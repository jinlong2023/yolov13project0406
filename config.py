"""
========================================================================
云台闭环目标追踪系统 - 全局配置文件
========================================================================
修改记录:
- GimbalConfig 新增: simulate_mode, gcu_ip, comm_mode, track_mode
- CameraConfig 新增: rtsp 模式自动配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DetectorConfig:
    """目标检测模块配置 (YOLOv13)"""
    model_name: str = "E:/yolov13project/yolov13/runs/detect/train3/weights/best.pt"
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.5
    input_size: Tuple[int, int] = (320, 320)
    target_classes: List[int] = field(default_factory=lambda: [0])
    device: str = "cuda:0"
    half_precision: bool = True
    max_det: int = 50
    augment: bool = False


@dataclass
class TrackerConfig:
    """目标跟踪模块配置"""
    process_noise_std: float = 5.0
    measurement_noise_std: float = 0.5
    dt: float = 1.0          # 单位: 帧 (速度=像素/帧)
    max_age: int = 90
    min_hits: int = 3
    iou_threshold: float = 0.05
    feature_dim: int = 128
    feature_weight: float = 0.6
    reid_threshold: float = 0.45
    optical_flow_winsize: int = 15
    optical_flow_maxlevel: int = 3
    velocity_smooth_factor: float = 0.7
    occlusion_threshold: float = 0.3
    prediction_only_frames: int = 90


@dataclass
class GimbalConfig:
    """云台PID控制模块配置"""
    # ===== PID 参数 =====
    kp_yaw: float = 3.0
    ki_yaw: float = 0.05
    kd_yaw: float = 1.5
    kp_pitch: float = 2.5
    ki_pitch: float = 0.04
    kd_pitch: float = 1.0
    max_yaw_speed: float = 180.0
    max_pitch_speed: float = 120.0
    max_yaw_angle: float = 170.0
    max_pitch_angle: float = 90.0
    min_pitch_angle: float = -30.0
    dead_zone: float = 0.0
    integral_limit: float = 50.0
    output_smooth_factor: float = 0.4
    control_frequency: float = 50.0
    prediction_enabled: bool = True
    prediction_horizon: int = 5
    compensation_factor: float = 0.5

    # ===== 模式选择 =====
    simulate_mode: bool = True       # True=仿真, False=Z-2Mini 硬件

    # ===== Z-2Mini 硬件参数 (simulate_mode=False 时生效) =====
    gcu_ip: str = "192.168.144.108"  # GCU IP 地址
    comm_mode: str = "udp"           # "udp" 或 "tcp"
    track_mode: str = "software_pid"
    # track_mode 选项:
    #   "software_pid"   - 上位机 PID 持续控制角速度 (原有逻辑)
    #   "gimbal_builtin" - YOLO 检测框 → 0x17, 吊舱内置跟踪 (推荐)


@dataclass
class CameraConfig:
    """相机参数配置"""
    source: str = "0"                # "0"=摄像头, 视频路径, 或 "rtsp://..."
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    fov_horizontal: float = 60.0
    fov_vertical: float = 34.0


@dataclass
class SystemConfig:
    """系统全局配置"""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    project_root: str = os.path.dirname(os.path.abspath(__file__))
    log_dir: str = os.path.join(project_root, "logs")
    result_dir: str = os.path.join(project_root, "results")
    data_dir: str = os.path.join(project_root, "data")

    show_display: bool = True
    show_trajectory: bool = True
    show_info_panel: bool = True
    display_width: int = 1920
    display_height: int = 1080
    record_video: bool = False
    output_video_path: str = "results/output.mp4"

    log_level: str = "INFO"
    save_tracking_data: bool = True
    save_interval: int = 100

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


def get_default_config() -> SystemConfig:
    """默认配置 (仿真模式, 摄像头)"""
    return SystemConfig()


def get_simulation_config() -> SystemConfig:
    """仿真测试配置"""
    config = SystemConfig()
    config.camera.source = "0"
    config.gimbal.simulate_mode = True
    config.detector.device = "cpu"
    return config


def get_z2mini_config() -> SystemConfig:
    """
    ★ Z-2Mini 真实硬件配置

    使用前请确保:
    1. 上位机 IP 设置为 192.168.144.x (与 GCU 同子网)
    2. ETH 线已连接 Z-2Mini
    3. Z-2Mini 已上电并完成初始化
    """
    config = SystemConfig()

    # 视频源: Z-2Mini RTSP 流
    config.camera.source = "rtsp://192.168.144.108"  # :554/"
    config.camera.frame_width = 1920
    config.camera.frame_height = 1080
    config.camera.fps = 30

    # 云台: 真实硬件模式
    config.gimbal.simulate_mode = False
    config.gimbal.gcu_ip = "192.168.144.108"
    config.gimbal.comm_mode = "udp"
    config.gimbal.track_mode = "gimbal_builtin"  # 推荐

    # GPU 推理
    config.detector.device = "cuda:0"
    config.detector.confidence_threshold = 0.4

    return config
