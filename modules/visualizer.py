"""
========================================================================
可视化与分析模块
========================================================================
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import time
import json
import os
from typing import List, Tuple, Optional, Dict
from collections import deque


class Visualizer:
    """实时可视化叠加"""

    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    ]

    def __init__(self, frame_size=(1280, 720)):
        self.frame_width, self.frame_height = frame_size
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

    def draw_frame(self, frame, trackers=None, primary_target=None,
                   control_output=None, gimbal_state=None, detections=None,
                   show_trajectory=True, show_info=True, detector_ok=True):
        display = frame

        # 画面中心十字线
        cx, cy = self.frame_width // 2, self.frame_height // 2
        cv2.line(display, (cx - 30, cy), (cx + 30, cy), (0, 200, 200), 1)
        cv2.line(display, (cx, cy - 30), (cx, cy + 30), (0, 200, 200), 1)
        cv2.circle(display, (cx, cy), 5, (0, 200, 200), 1)

        # ===== 检测框 (带类别名+置信度) =====
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det.bbox.astype(int)
                color = self.COLORS[det.class_id % len(self.COLORS)]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (x1, y1 - sz[1] - 8), (x1 + sz[0] + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                dcx, dcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(display, (dcx, dcy), 3, color, -1)

        # ===== 无检测警告 =====
        if not detections:
            h, w = display.shape[:2]
            ov = display.copy()
            cv2.rectangle(ov, (w // 2 - 200, 50), (w // 2 + 200, 100), (0, 0, 180), -1)
            cv2.addWeighted(ov, 0.7, display, 0.3, 0, display)
            msg = "YOLO NOT LOADED" if not detector_ok else "NO TARGET DETECTED"
            cv2.putText(display, msg, (w // 2 - 170, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ===== 跟踪框 =====
        if trackers:
            for trk in trackers:
                is_primary = primary_target and trk.id == primary_target.id
                color = (0, 255, 0) if is_primary else (255, 255, 0)
                bbox = trk.current_bbox.astype(int)
                x1, y1, x2, y2 = bbox
                thick = 3 if is_primary else 2
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)

                # ID标签
                label = f"ID:{trk.id}"
                if is_primary: label += " [TARGET]"
                if trk.is_occluded: label += " [OCC]"
                label += f" v={trk.current_speed:.1f}"
                sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x1, y1 - sz[1] - 8), (x1 + sz[0] + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                tcx, tcy = int(trk.current_center[0]), int(trk.current_center[1])
                cv2.circle(display, (tcx, tcy), 4, color, -1)

                # 运动轨迹
                if show_trajectory and len(trk.trajectory) > 1:
                    pts = list(trk.trajectory)
                    for i in range(1, len(pts)):
                        a = i / len(pts)
                        c = tuple(int(v * a) for v in color)
                        cv2.line(display, (int(pts[i-1][0]), int(pts[i-1][1])),
                                 (int(pts[i][0]), int(pts[i][1])), c, 2)

                # 预测轨迹 (主目标)
                if is_primary:
                    future = trk.predict_future(10)
                    if future:
                        cv2.arrowedLine(display, (tcx, tcy),
                                        (int(future[-1][0]), int(future[-1][1])),
                                        (0, 128, 255), 1, tipLength=0.3)

        # 主目标到中心连线
        if primary_target:
            tcx, tcy = primary_target.current_center
            fcx, fcy = self.frame_width // 2, self.frame_height // 2
            cv2.line(display, (int(tcx), int(tcy)), (fcx, fcy), (0, 0, 255), 1)
            dist = np.sqrt((tcx - fcx)**2 + (tcy - fcy)**2)
            mx, my = int((tcx + fcx) / 2), int((tcy + fcy) / 2)
            cv2.putText(display, f"{dist:.0f}px", (mx + 5, my - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 控制矢量
        if control_output:
            ecx, ecy = self.frame_width // 2, self.frame_height // 2
            ex = int(ecx + control_output.yaw_cmd * 2)
            ey = int(ecy + control_output.pitch_cmd * 2)
            cv2.arrowedLine(display, (ecx, ecy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)

        # 锁定指示
        if control_output and control_output.is_locked:
            h, w = display.shape[:2]
            m, s = 50, 30
            g = (0, 255, 0)
            cv2.line(display, (m, m), (m+s, m), g, 3)
            cv2.line(display, (m, m), (m, m+s), g, 3)
            cv2.line(display, (w-m, m), (w-m-s, m), g, 3)
            cv2.line(display, (w-m, m), (w-m, m+s), g, 3)
            cv2.line(display, (m, h-m), (m+s, h-m), g, 3)
            cv2.line(display, (m, h-m), (m, h-m-s), g, 3)
            cv2.line(display, (w-m, h-m), (w-m-s, h-m), g, 3)
            cv2.line(display, (w-m, h-m), (w-m, h-m-s), g, 3)
            cv2.putText(display, "LOCKED", (w//2 - 50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, g, 2)

        # ===== 信息面板 =====
        if show_info:
            self._draw_info(display, trackers, primary_target, control_output,
                            gimbal_state, detections, detector_ok)

        return display

    def _draw_info(self, frame, trackers, primary, ctrl, gimbal, dets, det_ok):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        self.fps_history.append(1.0 / dt if dt > 0 else 0)
        avg_fps = np.mean(list(self.fps_history))

        pw, ph = 310, 260
        # 优化：仅对左上角 310x260 的小区域进行半透明矩阵运算，极大地节省 CPU 耗时！
        roi = frame[10:10 + ph, 10:10 + pw]
        black_rect = np.zeros_like(roi)
        cv2.addWeighted(black_rect, 0.6, roi, 0.4, 0, roi)
        frame[10:10 + ph, 10:10 + pw] = roi

        y, lh = 30, 22
        f, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
        W, G, Y, R = (255,255,255), (0,255,0), (0,255,255), (0,0,255)

        def put(text, color=W):
            nonlocal y
            cv2.putText(frame, text, (20, y), f, fs, color, 1)
            y += lh

        put(f"FPS: {avg_fps:.1f}", G)
        put(f"YOLO: {'OK' if det_ok else 'NOT LOADED!'}", G if det_ok else R)

        nd = len(dets) if dets else 0
        put(f"Detections: {nd}", G if nd > 0 else R)

        nt = len(trackers) if trackers else 0
        put(f"Active Tracks: {nt}", G if nt > 0 else Y)

        if primary:
            put(f"Primary: ID {primary.id}", Y)
            cx, cy = primary.current_center
            put(f"  Pos: ({cx:.0f}, {cy:.0f})")
            put(f"  Speed: {primary.current_speed:.1f} px/f")

        if gimbal:
            put(f"Gimbal Yaw: {gimbal.yaw:.1f} deg")
            put(f"Gimbal Pitch: {gimbal.pitch:.1f} deg")

        if ctrl:
            locked = "YES" if ctrl.is_locked else "NO"
            put(f"Locked: {locked}", G if ctrl.is_locked else R)
            err = np.sqrt(ctrl.yaw_error**2 + ctrl.pitch_error**2)
            put(f"Error: {err:.1f} px")


class DataRecorder:
    """数据记录器"""
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = []

    def record_frame(self, frame_id, detections=None, primary_target=None,
                     control_output=None, gimbal_state=None, detect_time=0.0):
        rec = {'frame_id': frame_id, 'timestamp': time.time(),
               'detect_time_ms': detect_time * 1000,
               'num_detections': len(detections) if detections else 0}


        primary_conf = 0.0
        if detections and primary_target:
            for det in detections:
                if self._iou(det.bbox, primary_target.current_bbox) > 0.5:
                    primary_conf = det.confidence
                    break

        if primary_target:
            cx, cy = primary_target.current_center
            vx, vy = primary_target.current_velocity


            rec.update({
                'target_id': primary_target.id,
                'target_x': cx,
                'target_y': cy,
                'target_vx': vx,
                'target_vy': vy,
                'target_speed': primary_target.current_speed,
                'is_occluded': primary_target.is_occluded,
                'confidence': float(primary_conf),  # 记录置信度
                'age': primary_target.age,  # 记录生存周期
                'time_since_update': primary_target.time_since_update  # 记录盲推帧数
            })

        if control_output:
            rec.update({'error_x': control_output.yaw_error, 'error_y': control_output.pitch_error,
                        'yaw_cmd': control_output.yaw_cmd, 'pitch_cmd': control_output.pitch_cmd,
                        'is_locked': control_output.is_locked})
        if gimbal_state:
            rec.update({'gimbal_yaw': gimbal_state.yaw, 'gimbal_pitch': gimbal_state.pitch})

        self.data.append(rec)

    # 简单写一个私有 IoU 函数用于匹配当前框的置信度
    def _iou(self, b1, b2):
        x0 = max(b1[0], b2[0]);
        y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2]);
        y1 = min(b1[3], b2[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
        return inter / (union + 1e-6)

    def save(self, filename="tracking_data.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"[DataRecorder] 已保存: {path}")

    def generate_report(self):
        """
        生成系统性能评估报告 (量化指标增强版)
        论文核心点：引入 RMSE (均方根误差) 和 Standard Deviation (标准差)，
        用于量化评估系统在多场景下的跟踪误差、响应延迟、鲁棒性与稳定性。
        """
        if not self.data: return {}
        d = self.data
        n = len(d)

        # 提取各个维度的数据
        det_ms = [r.get('detect_time_ms', 0) for r in d]
        tracked = sum(1 for r in d if 'target_id' in r)
        errs = [np.sqrt(r['error_x'] ** 2 + r['error_y'] ** 2) for r in d if 'error_x' in r]
        locked = sum(1 for r in d if r.get('is_locked', False))

        # 🌟 学术点：计算 RMSE (均方根误差) 和 误差标准差 (表征系统稳定性/抖动量)
        avg_error = np.mean(errs) if errs else 0
        rmse = np.sqrt(np.mean(np.square(errs))) if errs else 0
        err_std = np.std(errs) if errs else 0

        report = {
            'total_frames': n,
            'detection': {
                'avg_ms': round(np.mean(det_ms), 2) if det_ms else 0,
                'fps': round(1000 / np.mean(det_ms), 1) if det_ms and np.mean(det_ms) > 0 else 0
            },
            'tracking': {
                'rate': round(tracked / n * 100, 1) if n > 0 else 0,
                'frames': tracked
            },
            'control': {
                'avg_error_px': round(avg_error, 2),
                'rmse_error_px': round(rmse, 2),  # 论文必备图表数据 (系统精度)
                'error_std_px': round(err_std, 2),  # 稳定性指标 (抗抖动能力)
                'lock_rate': round(locked / n * 100, 1) if n > 0 else 0
            }
        }

        # 保存到本地 JSON 文件，方便后续读取画图
        rp = os.path.join(self.save_dir, "performance_report.json")
        with open(rp, 'w') as f:
            json.dump(report, f, indent=2)

        # 在终端打印出极其专业的量化评估结果
        print(f"\n{'=' * 55}")
        print(f"  [系统性能评估报告 / Performance Report]")
        print(f"  总帧数 (Frames): {n}  |  跟踪率 (Track Rate): {report['tracking']['rate']}%")
        print(f"  锁定率 (Lock Rate): {report['control']['lock_rate']}%")
        print(f"  -------------------------------------------------------")
        print(f"  平均误差 (Avg Error):   {report['control']['avg_error_px']} px")
        print(f"  均方根误差 (RMSE):      {report['control']['rmse_error_px']} px  <-- (表征整体跟踪精度)")
        print(f"  误差标准差 (Std Dev):   {report['control']['error_std_px']} px  <-- (表征系统抗抖动稳定性)")
        print(f"{'=' * 55}\n")

        return report


def generate_analysis_plots(data_path, save_dir="results"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import platform
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
        elif platform.system() == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        plt.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("[WARNING] matplotlib not installed, skip plots")
        return

    with open(data_path) as f:
        data = json.load(f)
    frames = [d['frame_id'] for d in data]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gimbal Tracking Analysis', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    errs = [np.sqrt(d.get('error_x', 0)**2 + d.get('error_y', 0)**2) for d in data]
    ax.plot(frames, errs, 'b-', lw=0.8, alpha=0.7)
    if errs: ax.axhline(np.mean(errs), color='r', ls='--', label=f'Mean: {np.mean(errs):.1f}')
    ax.set_title('Tracking Error'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(frames, [d.get('gimbal_yaw', 0) for d in data], 'r-', lw=0.8, label='Yaw')
    ax.plot(frames, [d.get('gimbal_pitch', 0) for d in data], 'b-', lw=0.8, label='Pitch')
    ax.set_title('Gimbal Angle'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(frames, [d.get('yaw_cmd', 0) for d in data], 'r-', lw=0.8, label='Yaw')
    ax.plot(frames, [d.get('pitch_cmd', 0) for d in data], 'b-', lw=0.8, label='Pitch')
    ax.set_title('PID Output'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frames, [d.get('target_speed', 0) for d in data], 'g-', lw=0.8)
    ax.set_title('Target Speed'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, 'analysis_plots.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"[Plot] Saved: {p}")
