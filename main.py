"""
========================================================================
云台闭环目标追踪系统 - 主流程 (基于 YOLOv13)
========================================================================
前提条件:
  1. 从 https://github.com/iMoonLab/yolov13 克隆并安装 YOLOv13
  2. 下载 yolov13n.pt 权重文件到本项目根目录
  3. 运行: python main.py                (仿真模式, 摄像头)
          python main.py --z2mini        (Z-2Mini 硬件模式)
          python main.py --source video.mp4 (视频文件)

快捷键:
  q=退出  空格=暂停  r=重置  t=轨迹  i=信息面板  鼠标点击=选择目标
  p=拍照(硬件)  v=录像(硬件)  h=回中(硬件)
"""

import sys, os, time, argparse, threading
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (get_default_config,
                    get_z2mini_config)
from modules.detector import YOLODetector as TargetDetector
from modules.tracker import MultiObjectTracker
from modules.gimbal_controller import GimbalController
from modules.visualizer import Visualizer, DataRecorder, generate_analysis_plots
from performance_evaluator import PerformanceLogger

logger = PerformanceLogger("tracking_log.csv")

# ====================================================================
# RTSP 低延迟取流器  (新增)
# ====================================================================
class RTSPCapture:
    """
    RTSP 视频流线程化采集器

    OpenCV 默认的 VideoCapture.read() 从内部缓冲区读取, 对 RTSP 流
    会产生几帧延迟。此类使用独立线程持续取帧, 保证 read() 始终返回最新帧。
    """

    def __init__(self, url: str, width: int = 1920, height: int = 1080):
        self.url = url
        self.width = width
        self.height = height
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frame_count = 0

    def open(self) -> bool:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            print(f"[RTSP] ✗ 无法打开: {self.url}")
            return False

        print("[RTSP] 等待视频流稳定...")
        import time as _time
        deadline = _time.time() + 8.0
        got = False
        while _time.time() < deadline:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if w > 0 and h > 0:
                    self.width  = w
                    self.height = h
                    with self._lock:
                        self._frame = frame
                    got = True
                    break
            _time.sleep(0.1)

        if not got:
            print(f"[RTSP] ✗ 8秒内未收到有效帧: {self.url}")
            return False

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 1000:
            fps = 30.0
        print(f"[RTSP] 已连接: {self.width}x{self.height} @ {fps:.1f}fps")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_count += 1
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def isOpened(self):
        return self._running and self._cap is not None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.height
        if prop == cv2.CAP_PROP_FPS: return 30
        return 0


# ====================================================================
# 主系统
# ====================================================================
class GimbalTrackingSystem:
    """云台闭环目标追踪系统"""

    def __init__(self, config=None):
        self.config = config or get_default_config()
        c = self.config

        print("=" * 60)
        print("  云台闭环目标追踪系统 (YOLOv13)")
        if not c.gimbal.simulate_mode:
            print(f"  ★ Z-2Mini 硬件模式 | GCU: {c.gimbal.gcu_ip}")
            print(f"    跟踪: {c.gimbal.track_mode} | 通信: {c.gimbal.comm_mode}")
        else:
            print("  仿真模式")
        print("=" * 60)

        # 1. 目标检测
        print("\n[1/4] 初始化目标检测 (YOLOv13)...")
        self.detector = TargetDetector(c.detector)

        # 2. 多目标跟踪
        print("[2/4] 初始化跟踪模块 (Kalman + CNN)...")
        self.tracker = MultiObjectTracker(c.tracker)

        # 3. 云台控制
        print("[3/4] 初始化云台控制...")
        fsz = (c.camera.frame_width, c.camera.frame_height)
        self.gimbal_ctrl = GimbalController(c.gimbal, c.camera)

        # 4. 可视化
        print("[4/4] 初始化可视化...")
        self.visualizer = Visualizer(fsz)
        self.recorder = DataRecorder(c.result_dir)

        self.cap = None
        self.frame_count = 0
        self.is_running = False
        self.is_paused = False
        self.show_trajectory = c.show_trajectory
        self.show_info = c.show_info_panel
        self.mouse_click_pos = None
        self.video_writer = None

        if self.detector.model is None:
            print("\n" + "!" * 60)
            print("  警告: YOLOv13 模型未加载成功!")
            print("!" * 60 + "\n")
        else:
            print(f"\n[System] ✓ 系统就绪! 模型: {c.detector.model_name}\n")

    def _open_video(self):
        """打开视频源 (摄像头 / 视频文件 / RTSP)"""
        src = self.config.camera.source

        if src.startswith("rtsp://") or src.startswith("rtsps://"):
            # ===== RTSP 模式 (Z-2Mini) =====
            print(f"[Camera] RTSP 流: {src}")
            self.cap = RTSPCapture(src,
                                   self.config.camera.frame_width,
                                   self.config.camera.frame_height)
            if not self.cap.open():
                return False

        elif src.isdigit():
            # ===== 摄像头 =====
            self.cap = cv2.VideoCapture(int(src))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            print(f"[Camera] 摄像头 {src}")
        else:
            # ===== 视频文件 =====
            if not os.path.exists(src):
                print(f"[ERROR] 文件不存在: {src}")
                return False
            self.cap = cv2.VideoCapture(src)
            print(f"[Camera] 视频: {src}")

        if not self.cap.isOpened():
            print("[ERROR] 无法打开视频源!")
            return False

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] {w}x{h} @ {fps:.1f}fps")

        # 更新实际分辨率
        self.config.camera.frame_width = w
        self.config.camera.frame_height = h
        self.gimbal_ctrl = GimbalController(self.config.gimbal, self.config.camera)
        self.visualizer = Visualizer((w, h))

        if self.config.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config.output_video_path, fourcc, fps or 30, (w, h))
        return True

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click_pos = (x, y)

    def _handle_mouse(self):
        if self.mouse_click_pos is None:
            return
        mx, my = self.mouse_click_pos
        self.mouse_click_pos = None
        scale_x = self.config.camera.frame_width / self.config.display_width
        scale_y = self.config.camera.frame_height / self.config.display_height
        mx_orig = mx * scale_x
        my_orig = my * scale_y
        for t in self.tracker.get_active_trackers():
            b = t.current_bbox
            if b[0] <= mx_orig <= b[2] and b[1] <= my_orig <= b[3]:
                self.tracker.set_primary_target(t.id)
                print(f"[User] 选择目标 ID {t.id}")
                return

    def process_frame(self, frame):
        """核心流水线: 检测 → 跟踪 → 控制"""
        self.frame_count += 1
        t0 = time.time()

        # Step 1: YOLOv13 检测
        detections = self.detector.detect(frame)

        # Step 2: 卡尔曼+CNN 多目标跟踪
        active_trackers = self.tracker.update(detections)

        # Step 3: 主目标选择
        self._handle_mouse()
        if self.tracker.primary_target_id is None and active_trackers:
            fc = (self.config.camera.frame_width / 2,
                  self.config.camera.frame_height / 2)
            self.tracker.auto_select_primary(fc)
        primary = self.tracker.get_primary_target()

        # Step 4: 云台控制
        # 🌟 优化一：直接把整个 primary (主目标对象) 传给控制器，以便获取速度做前馈预测
        ctrl_dict = self.gimbal_ctrl.compute_control(
            primary, time.time()
        )

        if not self.config.gimbal.simulate_mode and self.gimbal_ctrl.commander:
            if self.config.gimbal.track_mode == "software_pid":
                self.gimbal_ctrl._software_pid(ctrl_dict)

        # 获取云台状态 (兼容你上一步的修复)
        gstate = self.gimbal_ctrl.get_gimbal_state() if hasattr(self.gimbal_ctrl, 'get_gimbal_state') else (
            self.gimbal_ctrl.gimbal.get_status() if hasattr(self.gimbal_ctrl,
                                                            'gimbal') and self.gimbal_ctrl.gimbal else None)

        # ====== 新增：数据转换器 (字典转对象) ======
        class CtrlObject:
            def __init__(self, d):
                self.yaw_cmd = d.get('yaw_speed', 0)
                self.pitch_cmd = d.get('pitch_speed', 0)
                self.yaw_error = d.get('error_x', 0)
                self.pitch_error = d.get('error_y', 0)
                self.is_locked = d.get('has_target', False)

        ctrl_obj = CtrlObject(ctrl_dict) if ctrl_dict else None
        # ============================================

        fps = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 30.0
        state = 0  # 默认 0 = Lost (丢失)
        err_x, err_y, cmd_y, cmd_p = 0, 0, 0, 0

        if primary is not None:
            # 如果有主目标，且视觉锁定了，记为 1；否则说明在进行卡尔曼盲推，记为 2
            state = 2 if primary.time_since_update > 0 else 1
            #state = 1 if (ctrl_obj and ctrl_obj.is_locked) else 2

        if ctrl_obj:
            err_x, err_y = ctrl_obj.yaw_error, ctrl_obj.pitch_error
            cmd_y, cmd_p = ctrl_obj.yaw_cmd, ctrl_obj.pitch_cmd

            # 调用全局 logger 记录当前帧的数据
        global logger
        logger.log_frame(err_x, err_y, cmd_y, cmd_p, state, fps)

        # Step 5: 可视化 (传入转换后的 ctrl_obj)
        display = self.visualizer.draw_frame(
            frame, active_trackers, primary, ctrl_obj, gstate, detections,
            self.show_trajectory, self.show_info, self.detector.model is not None)

        # Step 6: 记录 (传入转换后的 ctrl_obj)
        if self.config.save_tracking_data:
            self.recorder.record_frame(self.frame_count, detections, primary,
                                       ctrl_obj, gstate, time.time() - t0)

        if self.video_writer:
            self.video_writer.write(display)

        return display

    def run(self):
        if not self._open_video():
            return

        self.is_running = True
        wname = "YOLOv13 Gimbal Tracking"

        if self.config.show_display:
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(wname, self.config.display_width, self.config.display_height)
            cv2.moveWindow(wname, 0, 0)
            cv2.setMouseCallback(wname, self._mouse_cb)

        keys_hint = "q=退出 空格=暂停 r=重置 t=轨迹 i=面板 鼠标=选目标"
        if not self.config.gimbal.simulate_mode:
            keys_hint += " | p=拍照 v=录像 h=回中"
        print(f"\n[Run] {keys_hint}\n")

        try:
            while self.is_running:
                if self.is_paused:
                    key = cv2.waitKey(100) & 0xFF
                    self._key(key)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    # RTSP 模式下可能暂时失败, 重试
                    if self.config.camera.source.startswith("rtsp"):
                        time.sleep(0.01)
                        continue
                    print("[System] 视频结束")
                    break

                display = self.process_frame(frame)

                if self.config.show_display:
                    h, w = display.shape[:2]
                    dw, dh = self.config.display_width, self.config.display_height
                    if w != dw or h != dh:
                        display = cv2.resize(display, (dw, dh))
                    cv2.imshow(wname, display)

                key = cv2.waitKey(1) & 0xFF
                self._key(key)

                if (self.config.save_tracking_data and
                        self.frame_count % self.config.save_interval == 0):
                    self.recorder.save()

        except KeyboardInterrupt:
            print("\n[中断]")
        finally:
            self._cleanup()

    def _key(self, k):
        if k == ord('q') or k == 27:
            self.is_running = False
        elif k == ord(' '):
            self.is_paused = not self.is_paused
            print(f"[{'暂停' if self.is_paused else '继续'}]")
        elif k == ord('r'):
            self.tracker = MultiObjectTracker(self.config.tracker)
            self.gimbal_ctrl.reset()
            print("[重置]")
        elif k == ord('t'):
            self.show_trajectory = not self.show_trajectory
        elif k == ord('i'):
            self.show_info = not self.show_info
        # ★ 硬件模式额外快捷键
        elif k == ord('p'):
            self.gimbal_ctrl.take_photo()
            print("[拍照]")
        elif k == ord('v'):
            self.gimbal_ctrl.toggle_record()
            print("[录像切换]")
        elif k == ord('h'):
            self.gimbal_ctrl.reset()
            print("[回中]")
        elif k == ord('m'):
            if hasattr(self.gimbal_ctrl, 'commander') and self.gimbal_ctrl.commander:
                self.gimbal_ctrl.commander.toggle_pip(0x00)  # 0x00代表循环切换模式
                print("[相机] 切换画中画/红外视图")

    def _cleanup(self):
        # 停止吊舱跟踪
        if hasattr(self.gimbal_ctrl, 'gimbal') and hasattr(self.gimbal_ctrl.gimbal, 'disconnect'):
            self.gimbal_ctrl.gimbal.disconnect()

        global logger
        logger.close()

        if self.config.save_tracking_data:
            self.recorder.save()
            self.recorder.generate_report()
            dp = os.path.join(self.config.result_dir, "tracking_data.json")
            if os.path.exists(dp):
                generate_analysis_plots(dp, self.config.result_dir)

        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"  Frames: {self.frame_count}")
        s = self.detector.get_stats()
        print(f"  Detect FPS: {s['inference_fps']}  |  YOLO: {s['model_loaded']}")
        cs = self.gimbal_ctrl.get_stats()
        print(f"  Locked: {cs['is_locked']}  |  Quality: {cs['tracking_quality']:.2f}  |  Pitch: {cs['gimbal_pitch']:.1f}  Yaw: {cs['gimbal_yaw']:.1f}")
        if cs.get('is_hardware'):
            print(f"  Mode: {cs.get('gcu_mode','N/A')}  |  Zoom: {cs.get('gcu_zoom','N/A')}x")
        print(f"{'='*50}\n")


# ====================================================================
# CLI 入口
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv13 Gimbal Tracking")
    p.add_argument('--source', default=None,
                   help='0=摄像头, 视频路径, 或 rtsp://... (--z2mini 时自动设置)')
    p.add_argument('--model', default='yolov13n.pt',
                   help='YOLOv13模型 (yolov13n/s/l/x.pt)')
    p.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')
    p.add_argument('--conf', type=float, default=0.35, help='置信度阈值')
    p.add_argument('--target-class', type=int, nargs='*', default=None,
                   help='类别ID (不指定=所有, 0=人, 0 2 7=人+车+卡车)')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--record', action='store_true')
    p.add_argument('--display-size', type=int, nargs=2,
                   default=[1920, 1080], metavar=('W', 'H'))

    # PID 参数
    p.add_argument('--kp-yaw', type=float, default=0.8)
    p.add_argument('--ki-yaw', type=float, default=0.05)
    p.add_argument('--kd-yaw', type=float, default=0.15)
    p.add_argument('--kp-pitch', type=float, default=0.7)
    p.add_argument('--ki-pitch', type=float, default=0.04)
    p.add_argument('--kd-pitch', type=float, default=0.12)

    # ★ Z-2Mini 硬件参数
    p.add_argument('--z2mini', action='store_true',
                   help='启用 Z-2Mini 硬件模式 (自动设置 RTSP + GCU)')
    p.add_argument('--gcu-ip', default='192.168.144.108',
                   help='GCU IP 地址 (默认 192.168.144.108)')
    p.add_argument('--comm-mode', default='udp', choices=['udp', 'tcp'],
                   help='GCU 通信方式')
    p.add_argument('--track-mode', default='gimbal_builtin',
                   choices=['software_pid', 'gimbal_builtin'],
                   help='跟踪模式: gimbal_builtin(推荐) / software_pid')

    return p.parse_args()


def main():
    args = parse_args()

    # ★ 选择配置模板
    if args.z2mini:
        config = get_z2mini_config()
        print("[Config] 使用 Z-2Mini 硬件配置")
    else:
        config = get_default_config()

    # 覆盖参数
    if args.source is not None:
        config.camera.source = args.source
    elif args.z2mini and args.source is None:
        config.camera.source = f"rtsp://{args.gcu_ip}"

    config.detector.model_name = args.model
    config.detector.device = args.device
    config.detector.confidence_threshold = args.conf
    config.detector.target_classes = args.target_class if args.target_class else []
    config.show_display = not args.no_display
    config.record_video = args.record
    config.display_width = args.display_size[0]
    config.display_height = args.display_size[1]

    config.gimbal.kp_yaw = args.kp_yaw
    config.gimbal.ki_yaw = args.ki_yaw
    config.gimbal.kd_yaw = args.kd_yaw
    config.gimbal.kp_pitch = args.kp_pitch
    config.gimbal.ki_pitch = args.ki_pitch
    config.gimbal.kd_pitch = args.kd_pitch

    if args.z2mini:
        config.gimbal.simulate_mode = False
        config.gimbal.gcu_ip = args.gcu_ip
        config.gimbal.comm_mode = args.comm_mode
        config.gimbal.track_mode = args.track_mode

    system = GimbalTrackingSystem(config)
    system.run()


if __name__ == "__main__":
    main()
