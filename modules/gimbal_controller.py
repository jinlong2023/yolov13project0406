"""
云台控制器模块 (优化版 v2.0)
适配 SystemConfig / GimbalConfig / CameraConfig
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque

from gcu_protocol import GCUConnection, GCUCommander
from config import GimbalConfig, CameraConfig

# ── PID 控制器 ────────────────────────────────────────────────
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = 1000.0, integral_limit: float = 50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit  = output_limit
        self.integral_limit = integral_limit

        self.integral   = 0.0
        self.last_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        p_term = self.kp * error

        self.integral += error * dt
        self.integral  = np.clip(self.integral,
                                 -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        output = np.clip(p_term + i_term + d_term,
                         -self.output_limit, self.output_limit)
        self.last_error = error
        return output

    def reset(self):
        self.integral   = 0.0
        self.last_error = 0.0

# ── 云台控制器 ────────────────────────────────────────────────
class GimbalController:
    def __init__(self, gimbal_cfg: GimbalConfig,
                 camera_cfg: Optional[CameraConfig] = None):
        self.cfg = gimbal_cfg

        # 画面尺寸
        if camera_cfg:
            self.frame_w = camera_cfg.frame_width
            self.frame_h = camera_cfg.frame_height
        else:
            self.frame_w = 1280
            self.frame_h = 720
        self.frame_cx = self.frame_w / 2.0
        self.frame_cy = self.frame_h / 2.0

        # GCU 连接
        self.gimbal    = None
        self.commander = None
        if not gimbal_cfg.simulate_mode:
            self.gimbal = GCUConnection(gimbal_cfg)
            if self.gimbal.connect():
                self.commander = GCUCommander(self.gimbal)
                self.commander.set_pure_camera_mode()
            else:
                print("[Gimbal] ⚠ GCU 连接失败，退回仿真模式")
                self.cfg.simulate_mode = True


        # PID
        self.pid_yaw = PIDController(
            kp=gimbal_cfg.kp_yaw,
            ki=gimbal_cfg.ki_yaw,
            kd=gimbal_cfg.kd_yaw,
            output_limit=gimbal_cfg.max_yaw_speed,
            integral_limit=gimbal_cfg.integral_limit
        )

        self.pid_pitch = PIDController(
            kp=gimbal_cfg.kp_pitch,
            ki=gimbal_cfg.ki_pitch,
            kd=gimbal_cfg.kd_pitch,
            output_limit=gimbal_cfg.max_pitch_speed,
            integral_limit=gimbal_cfg.integral_limit
        )

        # 输出平滑
        self.smooth_yaw   = 0.0
        self.smooth_pitch = 0.0

        # 跟踪质量监控 (新增)
        self._tracking_quality_history = deque(maxlen=30)
        self._fallback_threshold       = 0.3

        # 动态 dt (新增)
        self._last_control_ts = time.time()

        # 连接健康监控 (新增)
        self._last_health_check    = time.time()
        self._health_check_interval = 2.0

        if not gimbal_cfg.simulate_mode and self.commander:
            # 刚连上时，先发一次 0x11 激活指向锁定模式，再发 0x00
            self.commander.conn._send_packet(0x11)
            time.sleep(0.2)
            self.commander.conn._send_packet(0x00)

        print(f"[Gimbal] 初始化完成 "
              f"(模式={gimbal_cfg.track_mode}, 仿真={gimbal_cfg.simulate_mode})")

    # ── 核心控制 ──────────────────────────────────────────────
    def compute_control(self,
                        target: Optional['KalmanTracker'],
                        timestamp: float = None) -> Dict:
        if timestamp is None:
            timestamp = time.time()

        # 动态 dt
        dt = timestamp - self._last_control_ts
        dt = max(0.01, min(dt, 0.5))
        self._last_control_ts = timestamp

        # 连接健康检查
        if getattr(self.cfg, 'simulate_mode', False) is False and getattr(self, 'gimbal', None):
            if timestamp - self._last_health_check > self._health_check_interval:
                if not self.gimbal.is_healthy():
                    print("[Gimbal] ⚠ 连接不健康，发送心跳恢复")
                    self.commander.heartbeat()
                self._last_health_check = timestamp

        if target is None:
            self.pid_yaw.reset()
            self.pid_pitch.reset()
            self.smooth_yaw = 0.0
            self.smooth_pitch = 0.0
            return {'yaw_speed': 0, 'pitch_speed': 0, 'has_target': False}

        target_position = target.current_center
        target_velocity = target.current_velocity

        if getattr(target, 'time_since_update', 0) > 10:
            self.pid_yaw.reset()
            self.pid_pitch.reset()
            self.smooth_yaw = 0.0
            self.smooth_pitch = 0.0
            return {'yaw_speed': 0, 'pitch_speed': 0, 'has_target': False}  # True 改成 False!

        # 无目标
        if target_position is None:
            self.pid_yaw.reset()
            self.pid_pitch.reset()
            self.smooth_yaw   = 0.0
            self.smooth_pitch = 0.0
            return {'yaw_speed': 0, 'pitch_speed': 0, 'has_target': False}

        # 误差计算
        error_x = target_position[0] - self.frame_cx
        error_y = target_position[1] - self.frame_cy

        # 归一化
        error_x_norm = error_x / (self.frame_w / 2.0)
        error_y_norm = error_y / (self.frame_h / 2.0)

        error_x_norm = np.clip(error_x_norm, -2.0, 2.0)
        error_y_norm = np.clip(error_y_norm, -2.0, 2.0)

        # 死区 (使用 GimbalConfig.dead_zone，单位°，这里按百分比折算)
        dz = self.cfg.dead_zone / 100.0
        if abs(error_x_norm) < dz:
            error_x_norm = 0.0
            self.pid_yaw.integral = 0.0  # 🌟 修复：进入死区后立刻清空累积的偏航漂移
        if abs(error_y_norm) < dz:
            error_y_norm = 0.0
            self.pid_pitch.integral = 0.0  # 🌟 修复：进入死区后立刻清空累积的俯仰下垂

        # 🌟 优化1：安全获取速度，防止 None 导致崩溃
        vx_norm = 0.0
        vy_norm = 0.0
        if target_velocity is not None:
            vx_norm = target_velocity[0] / (self.frame_w / 2.0)
            vy_norm = target_velocity[1] / (self.frame_h / 2.0)

        # 🌟 优化2：删除重复的 PID 计算，并将速度前馈真正加入到输出指令中
        yaw_pid_out = self.pid_yaw.compute(error_x_norm, dt)
        pitch_pid_out = self.pid_pitch.compute(-error_y_norm, dt)

        yaw_out   = yaw_pid_out + (vx_norm * self.cfg.compensation_factor)
        pitch_out = pitch_pid_out - (vy_norm * self.cfg.compensation_factor)

        # 平滑
        alpha = self.cfg.output_smooth_factor
        self.smooth_yaw   = alpha * self.smooth_yaw   + (1 - alpha) * yaw_out
        self.smooth_pitch = alpha * self.smooth_pitch + (1 - alpha) * pitch_out

        # 🌟 优化3：增加硬件安全锁。在乘以 100 之前限制基准值，防止云台协议溢出暴走
        MAX_BASE_SPEED = 100.0
        self.smooth_yaw = max(-MAX_BASE_SPEED, min(MAX_BASE_SPEED, self.smooth_yaw))
        self.smooth_pitch = max(-MAX_BASE_SPEED, min(MAX_BASE_SPEED, self.smooth_pitch))

        return {
            # ⚠️ 乘以 100，适配 GCU 的 0.1°/s 协议单位
            'yaw_speed': int(self.smooth_yaw * 100),
            'pitch_speed': int(self.smooth_pitch * 100),
            'has_target': True,
            'error_x': error_x,
            'error_y': error_y
        }

    def update(self,
               target_position: Optional[Tuple[float, float]],
               target_bbox: Optional[np.ndarray] = None,
               timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()

        output = self.compute_control(target_position, timestamp)

        if not self.cfg.simulate_mode and self.commander:
            if self.cfg.track_mode == "gimbal_builtin":
                self._builtin_tracking(target_bbox, output)
            else:
                self._software_pid(output)

    def _builtin_tracking(self, target_bbox: Optional[np.ndarray], output: Dict):
        if target_bbox is not None:
            x0 = int(target_bbox[0] / self.frame_w * 10000)
            y0 = int(target_bbox[1] / self.frame_h * 10000)
            x1 = int(target_bbox[2] / self.frame_w * 10000)
            y1 = int(target_bbox[3] / self.frame_h * 10000)

            result = self.commander.start_tracking(x0, y0, x1, y1)
            success = result is True
            self._tracking_quality_history.append(1 if success else 0)

            # 低质量警告
            if len(self._tracking_quality_history) >= 30:
                quality = sum(self._tracking_quality_history) / 30
                if quality < self._fallback_threshold:
                    print(f"[Gimbal] ⚠ 跟踪质量低 ({quality:.1%})，"
                          f"建议切换 track_mode='software_pid'")
        else:
            self.commander.stop_tracking()
            self._tracking_quality_history.append(0)

    def _software_pid(self, output: Dict):
        if output['has_target']:
            self.commander.pointing_lock(
                pitch_speed=output['pitch_speed'],
                yaw_speed=output['yaw_speed'],
                auto_zoom_compensate=True
            )
        else:
            self.commander.pointing_lock(pitch_speed=0, yaw_speed=0)

    # ── 工具方法 ──────────────────────────────────────────────
    def reset(self):
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.smooth_yaw   = 0.0
        self.smooth_pitch = 0.0
        self._tracking_quality_history.clear()
        if not self.cfg.simulate_mode and self.commander:
            self.commander.reset_gimbal()
        print("[Gimbal] 云台已复位")

    def get_tracking_quality(self) -> float:
        if not self._tracking_quality_history:
            return 1.0
        return sum(self._tracking_quality_history) / len(self._tracking_quality_history)

    def get_gimbal_state(self) -> Optional[Dict]:
        """获取底层云台的最新遥测状态"""
        if not self.cfg.simulate_mode and self.gimbal:
            return self.gimbal.get_status()
        return None

    def get_stats(self) -> Dict:
        status     = self.gimbal.get_status()    if self.gimbal    else None
        conn_stats = self.gimbal.get_stats()     if self.gimbal    else None
        return {
            'track_mode':       self.cfg.track_mode,
            'simulate_mode':    self.cfg.simulate_mode,
            'tracking_quality': self.get_tracking_quality(),
            'gimbal_pitch':     status.pitch          if status else 0.0,
            'gimbal_yaw':       status.yaw            if status else 0.0,
            'gimbal_zoom':      status.cam1_zoom       if status else 1.0,
            'is_locked':        status.is_tracking_ok if status else False,
            'conn_stats':       conn_stats
        }

    def shutdown(self):
        if not self.cfg.simulate_mode and self.commander:
            self.commander.stop_tracking()
            self.commander.pointing_lock(pitch_speed=0, yaw_speed=0)
            self.gimbal.disconnect()
        print("[Gimbal] 控制器已关闭")