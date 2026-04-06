"""
GCU 通信协议模块
仅保留基础云台 PID 控制与基础相机功能(拍照/录像/变倍)
新增：一键关闭 OSD 等干扰画面的内置辅助功能
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional
from config import GimbalConfig

# ── 协议常量 ──────────────────────────────────────────────────
PROTOCOL_HEADER_SEND = bytes([0xA8, 0xE5])
PROTOCOL_HEADER_RECV = bytes([0x8A, 0x5E])
PROTOCOL_VERSION     = 0x01
GCU_SEND_PORT        = 2337   # 发送到设备的端口
GCU_RECV_PORT        = 2338   # 本地绑定接收端口

@dataclass
class GimbalStatus:
    mode:           int
    pitch:          float
    yaw:            float
    roll:           float
    cam1_zoom:      float
    cam2_zoom:      float
    is_tracking_ok: bool
    target_x:       int
    target_y:       int

def _crc16(data: bytes) -> int:
    crc_ta = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
        0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef
    ]
    crc = 0
    for byte in data:
        da  = (crc >> 12) & 0x0F
        crc = (crc << 4) & 0xFFFF
        crc ^= crc_ta[da ^ ((byte >> 4) & 0x0F)]
        da  = (crc >> 12) & 0x0F
        crc = (crc << 4) & 0xFFFF
        crc ^= crc_ta[da ^ (byte & 0x0F)]
    return crc

# ── GCU 连接底层 ────────────────────────────────────────────────
class GCUConnection:
    FLAG_CONTROL_VALID   = 0x04
    FLAG_IMU_VALID       = 0x01

    def __init__(self, cfg: GimbalConfig):
        self.host = cfg.gcu_ip
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None

        self.latest_status: Optional[GimbalStatus] = None
        self.status_lock   = threading.Lock()
        self._recv_buffer  = bytearray()

        self._roll    = 0
        self._pitch   = 0
        self._yaw     = 0
        self._ctrl_ok = True

        self._total_recv      = 0
        self._crc_errors      = 0
        self._last_recv_time  = 0.0

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", GCU_RECV_PORT))
            self.sock.settimeout(1.0)
            self.running = True
            self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self.recv_thread.start()
            print(f"[GCU] 连接成功 (UDP {self.host}:{GCU_SEND_PORT})")
            return True
        except Exception as e:
            print(f"[GCU] 连接失败: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
        print("[GCU] 连接已断开")

    # ★ 修复了带参数发送的底层 BUG
    def _send_packet(self, command: int = 0x00, params: bytes = b"") -> bool:
        if not self.sock:
            return False
        try:
            pkt = self._build_packet(command, params)
            self.sock.sendto(pkt, (self.host, GCU_SEND_PORT))
            return True
        except Exception as e:
            return False

    def _build_packet(self, command: int = 0x00, params: bytes = b"") -> bytes:
        pkt = bytearray()
        packet_len = 72 + len(params)  # ★ 动态包长度

        pkt.extend(PROTOCOL_HEADER_SEND)             # 2字节 帧头
        pkt.extend(struct.pack("<H", packet_len))    # 2字节 包长度
        pkt.append(PROTOCOL_VERSION)                 # 1字节 版本号
        pkt.extend(struct.pack("<h", self._roll))    # 2字节 滚转控制量
        pkt.extend(struct.pack("<h", self._pitch))   # 2字节 俯仰控制量
        pkt.extend(struct.pack("<h", self._yaw))     # 2字节 偏航控制量

        flag = 0
        if self._ctrl_ok: flag |= self.FLAG_CONTROL_VALID
        flag |= self.FLAG_IMU_VALID
        pkt.append(flag)                             # 1字节 状态标志

        pkt.extend(bytes(18))                        # 18字节 载机数据 (必须是18!)
        pkt.append(0x01)                             # 1字节 请求副帧
        pkt.extend(bytes(6))                         # 6字节 预留
        pkt.append(0x01)                             # 1字节 副帧帧头
        pkt.extend(bytes(31))                        # 31字节 GNSS数据

        pkt.append(command)                          # 1字节 指令
        pkt.extend(params)                           # n字节 参数

        crc = _crc16(bytes(pkt))
        pkt.extend(struct.pack(">H", crc))           # 2字节 CRC
        return bytes(pkt)

    def set_control(self, pitch: int = 0, yaw: int = 0, valid: bool = True):
        self._pitch   = pitch
        self._yaw     = yaw
        self._ctrl_ok = valid

    def _recv_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                self._recv_buffer.extend(data)
                self._parse_buffer()
            except socket.timeout:
                continue
            except Exception as e:
                break

    def _parse_buffer(self):
        while len(self._recv_buffer) >= 72:
            idx = self._recv_buffer.find(PROTOCOL_HEADER_RECV)
            if idx == -1:
                self._recv_buffer.clear()
                return
            if idx > 0:
                self._recv_buffer = self._recv_buffer[idx:]
            if len(self._recv_buffer) < 4:
                return
            pkt_len = struct.unpack("<H", self._recv_buffer[2:4])[0]
            if len(self._recv_buffer) < pkt_len:
                return
            pkt = bytes(self._recv_buffer[:pkt_len])
            self._recv_buffer = self._recv_buffer[pkt_len:]
            self._parse_packet(pkt)

    def _parse_packet(self, pkt: bytes):
        if len(pkt) < 72: return
        self._total_recv += 1
        self._last_recv_time = time.time()

        crc_recv = struct.unpack(">H", pkt[-2:])[0]
        crc_calc = _crc16(pkt[:-2])
        if crc_recv != crc_calc:
            self._crc_errors += 1
            return

        try:
            work_mode = pkt[5]
            pitch     = struct.unpack("<h", pkt[20:22])[0] / 100.0
            yaw       = struct.unpack("<H", pkt[22:24])[0] / 100.0
            roll      = struct.unpack("<h", pkt[18:20])[0] / 100.0
            cam1_zoom = 1.0
            cam2_zoom = 1.0
            if len(pkt) >= 63:
                cam1_zoom = struct.unpack("<H", pkt[59:61])[0] / 10.0
                cam2_zoom = struct.unpack("<H", pkt[61:63])[0] / 10.0
            off_h = struct.unpack("<h", pkt[8:10])[0] / 10.0
            off_v = struct.unpack("<h", pkt[10:12])[0] / 10.0

            status = GimbalStatus(
                mode=work_mode, pitch=pitch, yaw=yaw, roll=roll,
                cam1_zoom=cam1_zoom, cam2_zoom=cam2_zoom,
                is_tracking_ok=(work_mode == 0x17),
                target_x=int(off_h), target_y=int(off_v)
            )
            with self.status_lock:
                self.latest_status = status
        except Exception:
            pass

    def get_status(self) -> Optional[GimbalStatus]:
        with self.status_lock:
            return self.latest_status

    def is_healthy(self, timeout: float = 2.0) -> bool:
        if not self._last_recv_time: return False
        return (time.time() - self._last_recv_time) < timeout

    def get_stats(self) -> dict:
        return {
            "total_recv":    self._total_recv,
            "crc_errors":    self._crc_errors,
            "error_rate":    self._crc_errors / max(self._total_recv, 1),
            "is_healthy":    self.is_healthy()
        }

# ── GCU 高级控制接口 ──────────────────────────────────────────
class GCUCommander:
    def __init__(self, connection: GCUConnection):
        self.conn = connection

    # ================= 1. 纯净画面模式 =================
    def set_pure_camera_mode(self):
        """
        [极其重要] 关闭吊舱可能会显示在画面上的自带UI
        为上位机 YOLO 算法提供无干扰的纯净视频流
        """
        print("[GCU] 正在关闭内置OSD与辅助识别，净化视频流...")
        self.conn._send_packet(0x73, bytes([0x00])) # 关 OSD
        time.sleep(0.05)
        self.conn._send_packet(0x75, bytes([0x00])) # 关 内置目标识别
        time.sleep(0.05)
        self.conn._send_packet(0x81, bytes([0x00])) # 关 连续测距十字丝
        time.sleep(0.05)
        self.conn._send_packet(0x2B, bytes([0x01, 0x00]))  # 0x00=关，0x01=开，0x02=自动
        time.sleep(0.05)
        print("[GCU] 视频流净化完成，YOLO 专属模式就绪。")

    # ================= 2. 基础云台控制 =================
    def heartbeat(self) -> Optional[GimbalStatus]:
        self.conn._send_packet(0x00)
        time.sleep(0.05)
        return self.conn.get_status()

    def pointing_lock(self, pitch_speed: int, yaw_speed: int,
                      auto_zoom_compensate: bool = True) -> bool:
        if auto_zoom_compensate:
            st = self.conn.get_status()
            if st and st.cam1_zoom > 1.0:
                pitch_speed = int(pitch_speed * st.cam1_zoom)
                yaw_speed   = int(yaw_speed   * st.cam1_zoom)
        self.conn.set_control(pitch=pitch_speed, yaw=yaw_speed, valid=True)
        return self.conn._send_packet(0x00)

    def reset_gimbal(self) -> bool:
        self.conn.set_control(0, 0, False)
        return self.conn._send_packet(0x03)

    # ================= 3. 基本相机控制 =================
    def take_photo(self, camera_id: int = 0x01) -> bool:
        """拍照 (参数 0x01 默认控制可见光相机)"""
        return self.conn._send_packet(0x20, bytes([camera_id]))

    def toggle_record(self, camera_id: int = 0x01) -> bool:
        """开始/停止录像"""
        return self.conn._send_packet(0x21, bytes([camera_id]))

    def zoom_in(self, camera_id: int = 0x01) -> bool:
        """连续放大画面"""
        return self.conn._send_packet(0x22, bytes([camera_id]))

    def zoom_out(self, camera_id: int = 0x01) -> bool:
        """连续缩小画面"""
        return self.conn._send_packet(0x23, bytes([camera_id]))

    def zoom_stop(self, camera_id: int = 0x01) -> bool:
        """停止画面缩放"""
        return self.conn._send_packet(0x24, bytes([camera_id]))

    def toggle_pip(self, mode: int = 0x00) -> bool:
        """
        切换画中画(PiP)模式
        参数: 0x00=循环切换，0x01~0x04=指定特定的画面布局
        """
        return self.conn._send_packet(0x74, bytes([mode]))