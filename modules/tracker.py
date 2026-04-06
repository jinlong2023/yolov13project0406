"""
多目标跟踪模块 (优化版 v2.0)
适配 TrackerConfig
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from typing import Optional, List, Tuple, Dict
from collections import deque
from scipy.optimize import linear_sum_assignment

from config import TrackerConfig

# ── 卡尔曼跟踪器 ──────────────────────────────────────────────
class KalmanTracker:
    """
    状态向量: [cx, cy, w, h, vx, vy, vw, vh, ax, ay]
    新增: 面积稳定性检测 / 遮挡标记 / 匀加速多步预测
    """
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, cfg: Optional[TrackerConfig] = None):
        KalmanTracker._id_counter += 1
        self.id  = KalmanTracker._id_counter
        self.cfg = cfg or TrackerConfig()

        self.state_dim = 10
        self.obs_dim   = 4
        self._init_kalman()

        cx, cy, w, h = self._bbox_to_xywh(bbox)
        self.x = np.zeros((self.state_dim, 1))
        self.x[0], self.x[1], self.x[2], self.x[3] = cx, cy, w, h

        self.age               = 0
        self.time_since_update = 0
        self.hit_streak        = 0
        self.hits              = 0

        # 遮挡检测 (新增)
        self.is_occluded    = False
        self.occlusion_count = 0
        self._area_history  = deque(maxlen=10)
        self._area_history.append(w * h)

        self.feature: Optional[np.ndarray] = None
        self.trajectory = deque(maxlen=50)
        self.trajectory.append((cx, cy))
        self.last_update_time = time.time()

    def _init_kalman(self):
        n, m, dt = self.state_dim, self.obs_dim, self.cfg.dt

        self.F = np.eye(n)
        self.F[0, 4] = dt;  self.F[1, 5] = dt
        self.F[2, 6] = dt;  self.F[3, 7] = dt
        self.F[0, 8] = 0.5 * dt * dt
        self.F[1, 9] = 0.5 * dt * dt
        self.F[4, 8] = dt;  self.F[5, 9] = dt

        self.H = np.zeros((m, n))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1

        q = self.cfg.process_noise_std ** 2
        r = self.cfg.measurement_noise_std ** 2
        self.Q = np.eye(n) * q
        self.Q[4:8, 4:8] *= 10
        self.Q[8:10, 8:10] *= 5
        self.R = np.eye(m) * r
        self.P = np.eye(n) * 10.0
        self.P[4:, 4:] *= 100

    @staticmethod
    def _bbox_to_xywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy = self.x[0, 0], self.x[1, 0]
        w,  h  = max(self.x[2, 0], 1), max(self.x[3, 0], 1)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1

        # 🌟 修复一：卡尔曼物理阻尼器 (防止状态爆炸)
        if self.time_since_update > 0:
            # 盲推时，每一帧让速度衰减 10% (模拟空气阻力)
            self.x[4, 0] *= 0.9  # vx
            self.x[5, 0] *= 0.9  # vy
            self.x[6, 0] *= 0.9  # vw
            self.x[7, 0] *= 0.9  # vh
            # 盲推时，瞬间清零加速度！不要让它二次方发散！
            self.x[8, 0] = 0.0  # ax
            self.x[9, 0] = 0.0  # ay

        return self._state_to_bbox()
        # 增加 confidence 参数
    def update(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None, confidence: float = 1.0):
        cx, cy, w, h = self._bbox_to_xywh(bbox)
        z = np.array([[cx], [cy], [w], [h]])

        # 🌟 自适应观测噪声：置信度越低，噪声 R 呈指数级放大，系统将更依赖卡尔曼的运动预测
        adaptive_factor = np.exp(1.0 - confidence)
        adaptive_R = self.R * adaptive_factor

        # 重新计算卡尔曼增益 K
        S = self.H @ self.P @ self.H.T + adaptive_R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

            # ... 后面的区域稳定判断和轨迹记录保持不变 ...

        self._area_history.append(w * h)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_update_time = time.time()

        if feature is not None:
            self.feature = feature if self.feature is None \
                else 0.9 * self.feature + 0.1 * feature

        self.trajectory.append((self.x[0, 0], self.x[1, 0]))

        if self.is_occluded:
            self.is_occluded    = False
            self.occlusion_count = 0

    def mark_occluded(self):
        self.is_occluded = True
        self.occlusion_count += 1
        self.hit_streak = 0

    def is_area_stable(self, threshold: float = 0.5) -> bool:
        if len(self._area_history) < 3:
            return True
        areas      = list(self._area_history)
        mean_area  = np.mean(areas[:-1])
        if mean_area < 1:
            return True
        return abs(areas[-1] - mean_area) / mean_area < threshold

    def predict_future(self, steps: int = 5) -> List[Tuple[float, float]]:
        cx = self.x[0, 0]; cy = self.x[1, 0]
        vx = self.x[4, 0]; vy = self.x[5, 0]
        ax = self.x[8, 0]; ay = self.x[9, 0]
        return [
            (cx + vx * t + 0.5 * ax * t * t,
             cy + vy * t + 0.5 * ay * t * t)
            for t in range(1, steps + 1)
        ]

    @property
    def current_bbox(self)   -> np.ndarray:          return self._state_to_bbox()
    @property
    def current_center(self) -> Tuple[float, float]: return (self.x[0,0], self.x[1,0])
    @property
    def current_velocity(self)-> Tuple[float, float]:return (self.x[4,0], self.x[5,0])
    @property
    def current_area(self)   -> float:
        return max(self.x[2,0], 1) * max(self.x[3,0], 1)
    @property
    def current_speed(self) -> float:
        """计算标量速度 (像素/帧)"""
        vx, vy = self.current_velocity
        return float(np.sqrt(vx ** 2 + vy ** 2))

# ── 多目标跟踪管理器 ──────────────────────────────────────────
class MultiObjectTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.trackers: List[KalmanTracker] = []
        self.primary_target_id: Optional[int] = None

        # 主目标超时 (新增)
        self._primary_missing_frames  = 0
        self._primary_timeout = getattr(cfg, 'max_age', 90)
        #self._primary_timeout         = getattr(cfg, 'prediction_only_frames', 15)

        self._frame_count      = 0
        self._total_detections = 0

    def update(self, detections: list) -> List[KalmanTracker]:
        self._frame_count      += 1
        self._total_detections += len(detections)

        for t in self.trackers:
            t.predict()

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for d_idx, t_idx in matched:
            det = detections[d_idx]
            self.trackers[t_idx].update(
                det.bbox,
                feature=getattr(det, 'feature', None),
                confidence = det.confidence
            )

        # 遮挡标记 (新增)
        for t_idx in unmatched_trks:
            if not self.trackers[t_idx].is_area_stable():
                self.trackers[t_idx].mark_occluded()

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            tk  = KalmanTracker(det.bbox, self.cfg)
            if getattr(det, 'feature', None) is not None:
                tk.feature = det.feature
            self.trackers.append(tk)

        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.cfg.max_age]

        self._check_primary_timeout()

        return [t for t in self.trackers
                if t.hit_streak >= self.cfg.min_hits or t.hits >= self.cfg.min_hits]

    def _check_primary_timeout(self):
        if self.primary_target_id is None:
            self._primary_missing_frames = 0
            return

        primary = next((t for t in self.trackers
                        if t.id == self.primary_target_id), None)

        if primary is None or primary.time_since_update > 0:
            self._primary_missing_frames += 1
        else:
            self._primary_missing_frames = 0

        if self._primary_missing_frames >= self._primary_timeout:
            print(f"[Tracker] 主目标 {self.primary_target_id} "
                  f"超时 ({self._primary_timeout}帧)，已清除")
            self.primary_target_id       = None
            self._primary_missing_frames = 0

    def _associate(self, detections: list) -> Tuple[List, List, List]:
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        iou_mat = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_mat[d, t] = self._iou(det.bbox, trk.current_bbox)

        if self.cfg.feature_weight > 0:
            feat_mat = np.zeros_like(iou_mat)
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    df = getattr(det, 'feature', None)
                    if df is not None and trk.feature is not None:
                        sim = np.dot(df, trk.feature) / (
                                np.linalg.norm(df) * np.linalg.norm(trk.feature) + 1e-6)
                        feat_mat[d, t] = max(0.0, sim)

            # 🌟 核心创新点：引入 ReID 特征门控惩罚机制 (对应毕设指标2)
            # 如果两个框的空间 IoU 很高(贴在一起)，但衣服颜色特征(HSV)极度不匹配，
            # 说明发生了交叉遮挡干扰！直接将代价矩阵清零，强制系统拒绝认错人！
            for d in range(len(detections)):
                for t in range(len(self.trackers)):
                    # 如果算出了特征相似度，但低于阈值 (比如 0.45)
                    if feat_mat[d, t] > 0 and feat_mat[d, t] < self.cfg.reid_threshold:
                        iou_mat[d, t] = 0.0  # 破坏其空间匹配逻辑
                        feat_mat[d, t] = 0.0  # 破坏其特征匹配逻辑

            cost = (1 - self.cfg.feature_weight) * iou_mat + \
                   self.cfg.feature_weight * feat_mat
        else:
            cost = iou_mat

        row_ind, col_ind = linear_sum_assignment(-cost)

        matched, unmatched_dets, unmatched_trks = \
            [], list(range(len(detections))), list(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            # 🌟 终极修复：增加 ReID 盲推空间漂移抢救机制
            # 提取当前的特征相似度分数
            current_feat_score = feat_mat[r, c] if self.cfg.feature_weight > 0 else 0.0

            # 如果特征相似度极高 (比如大于 0.75)，即使 IoU 为 0 (预测框漂移了)，也强制认定是同一个人！
            is_reid_rescue = False
            if current_feat_score >= getattr(self.cfg, 'reid_threshold', 0.7):
                is_reid_rescue = True

            # 匹配门槛：综合得分大于常规阈值，或者触发了硬性特征抢救
            if cost[r, c] >= self.cfg.iou_threshold or is_reid_rescue:
                matched.append((r, c))
                unmatched_dets.remove(r)
                unmatched_trks.remove(c)

                # 如果是抢救回来的（说明发生了瞬移），为了防止卡尔曼滤波被惯性拖慢，
                # 我们可以强制重置一下卡尔曼的物理坐标，让它瞬间“闪现”回来
                if is_reid_rescue and iou_mat[r, c] < 0.1:
                    bbox = detections[r].bbox
                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    self.trackers[c].x[0, 0] = cx
                    self.trackers[c].x[1, 0] = cy

       # for r, c in zip(row_ind, col_ind):
            # 匹配门槛：综合得分必须大于配置的 iou_threshold
            #if cost[r, c] >= self.cfg.iou_threshold:
                #matched.append((r, c))
                #unmatched_dets.remove(r)
                #unmatched_trks.remove(c)
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            for t in list(unmatched_trks):
                trk = self.trackers[t]

                # 规则1：只抢救正在盲推的目标 (time_since_update > 0)
                if trk.time_since_update > 0:
                    trk_cx, trk_cy = trk.current_center
                    best_d = -1
                    min_dist = float('inf')

                    # 💡 动态膨胀搜索区：盲推时间越长，允许的抢救半径越大 (基础100像素 + 每帧膨胀20像素)
                    max_allowed_radius = 100 + (trk.time_since_update * 20)

                    for d in unmatched_dets:
                        det = detections[d]
                        det_cx = (det.bbox[0] + det.bbox[2]) / 2
                        det_cy = (det.bbox[1] + det.bbox[3]) / 2

                        # 计算物理中心点欧氏距离
                        dist = np.sqrt((det_cx - trk_cx) ** 2 + (det_cy - trk_cy) ** 2)

                        if dist < min_dist and dist < max_allowed_radius:
                            min_dist = dist
                            best_d = d

                    # 规则2：如果在搜索圈内找到了最接近的目标，强制接管并归位！
                    if best_d != -1:
                        matched.append((best_d, t))
                        unmatched_dets.remove(best_d)
                        unmatched_trks.remove(t)

                        # 【极其关键】消除物理漂移后遗症：
                        # 1. 强制重置卡尔曼坐标到真实视觉位置，瞬间拉回！
                        bbox = detections[best_d].bbox
                        trk.x[0, 0] = (bbox[0] + bbox[2]) / 2
                        trk.x[1, 0] = (bbox[1] + bbox[3]) / 2
                        # 2. 瞬间清零卡尔曼的预测速度！防止云台因为过去的惯性而猛烈抽搐
                        trk.x[4, 0] = 0.0
                        trk.x[5, 0] = 0.0
                        print(f"[Tracker] 🚀 触发空间抢救! 主目标 {trk.id} 漂移 {min_dist:.1f} 像素被成功拉回!")

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
        x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
        inter = max(0, x1-x0) * max(0, y1-y0)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-6)

    def set_primary_target(self, target_id: int):
        self.primary_target_id       = target_id
        self._primary_missing_frames = 0
        print(f"[Tracker] 设置主目标 ID: {target_id}")

    def get_primary_target(self) -> Optional[KalmanTracker]:
        if self.primary_target_id is None:
            return None
        return next((t for t in self.trackers
                     if t.id == self.primary_target_id), None)

    def get_active_trackers(self) -> List[KalmanTracker]:
        """获取当前活跃的跟踪目标列表"""
        return [t for t in self.trackers
                if t.hit_streak >= self.cfg.min_hits or t.hits >= self.cfg.min_hits]

    def auto_select_primary(self, frame_center: Tuple[float, float]) \
            -> Optional[KalmanTracker]:
        active = [t for t in self.trackers if t.hit_streak >= self.cfg.min_hits]
        if not active:
            return None
        cx, cy = frame_center
        best = min(active, key=lambda t:
                   (t.current_center[0]-cx)**2 + (t.current_center[1]-cy)**2)
        self.set_primary_target(best.id)
        return best

    def get_stats(self) -> Dict:
        return {
            'frame_count':          self._frame_count,
            'active_tracks':        len([t for t in self.trackers
                                         if t.hit_streak >= self.cfg.min_hits]),
            'total_tracks':         len(self.trackers),
            'primary_target_id':    self.primary_target_id,
            'primary_missing_frames': self._primary_missing_frames,
            'avg_detections':       self._total_detections / max(self._frame_count, 1)
        }