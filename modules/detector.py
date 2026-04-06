"""
目标检测模块 (优化版 v2.0)
适配 DetectorConfig
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Detector] ⚠ ultralytics 未安装")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from config import DetectorConfig

@dataclass
class Detection:
    bbox:       np.ndarray
    confidence: float
    class_id:   int
    class_name: str
    feature:    Optional[np.ndarray] = field(default=None)

class YOLODetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg   = cfg
        self.model = None

        # 统计 (新增)
        self.frame_count          = 0
        self.error_count          = 0
        self._total_inference_ms  = 0.0
        self._total_detections    = 0
        self._start_time          = time.time()

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(self.cfg.model_name)
            print(f"[Detector] 模型加载成功: {self.cfg.model_name}")
        except Exception as e:
            print(f"[Detector] ⚠ 模型加载失败: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # 空帧保护 (新增)
        if frame is None or frame.size == 0:
            return []
        if not YOLO_AVAILABLE or self.model is None:
            return []

        self.frame_count += 1
        t0 = time.time()

        try:
            results = self.model(
                frame,
                conf=self.cfg.confidence_threshold,
                iou=self.cfg.nms_threshold,
                classes=self.cfg.target_classes if self.cfg.target_classes else None,
                imgsz=self.cfg.input_size[0],
                device=self.cfg.device,
                half=self.cfg.half_precision,
                max_det=self.cfg.max_det,
                augment=self.cfg.augment,
                verbose=False
            )
            detections = self._parse(results, frame)

            self._total_inference_ms += (time.time() - t0) * 1000
            self._total_detections   += len(detections)
            return detections

        except Exception as e:
            self.error_count += 1
            if self.error_count % 10 == 1:
                print(f"[Detector] ⚠ 检测错误 (累计 {self.error_count}): {e}")
            return []

    def _parse(self, results, frame: np.ndarray) -> List[Detection]:
        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            try:
                bbox       = box.xyxy[0].cpu().numpy().astype(np.float32)
                confidence = float(box.conf[0])
                class_id   = int(box.cls[0])
                class_name = self.model.names.get(class_id, str(class_id))
                feature    = None

                detections.append(Detection(
                    bbox=bbox, confidence=confidence,
                    class_id=class_id, class_name=class_name,
                    feature=feature
                ))
            except Exception:
                self.error_count += 1
        return detections

    def _extract_feature(self, frame: np.ndarray,
                         bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        特征提取模块 (抗光照干扰版)
        论文核心点：仅提取目标 HSV 颜色空间中的 H(色相) 和 S(饱和度) 直方图，
        彻底丢弃 V(明度) 通道，从而实现对光照剧烈变化、阴影遮挡的强鲁棒性。
        """
        try:
            x0, y0, x1, y1 = bbox.astype(int)
            # 1. 边界保护，防止越界报错
            x0 = max(0, x0);
            y0 = max(0, y0)
            x1 = min(frame.shape[1], x1);
            y1 = min(frame.shape[0], y1)
            if x1 <= x0 or y1 <= y0:
                return None

            # 2. 截取目标区域 (ROI)
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                return None

            # 3. 转换到 HSV 颜色空间
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 4. 提取色相 (H) 和饱和度 (S) 的直方图特征
            # H通道范围 [0, 180]，分成 16 个 bin
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            # S通道范围 [0, 256]，分成 16 个 bin (相比原来提高了精度)
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()

            # 5. 特征级联与 L2 归一化 (让特征向量具有尺度不变性)
            feat = np.concatenate([hist_h, hist_s]).astype(np.float32)
            norm = np.linalg.norm(feat)

            return feat / norm if norm > 0 else feat

        except Exception as e:
            # 失败时静默返回，不让整个系统崩溃
            return None

    def get_stats(self) -> dict:
        elapsed = time.time() - self._start_time
        return {
            'frame_count':      self.frame_count,
            'error_count':      self.error_count,
            'error_rate':       self.error_count / max(self.frame_count, 1),
            'inference_fps':    self.frame_count  / max(elapsed, 1e-6),
            'avg_inference_ms': self._total_inference_ms / max(self.frame_count, 1),
            'avg_detections':   self._total_detections   / max(self.frame_count, 1),
            'model_loaded':     self.model is not None
        }

    def reset_stats(self):
        self.frame_count         = 0
        self.error_count         = 0
        self._total_inference_ms = 0.0
        self._total_detections   = 0
        self._start_time         = time.time()