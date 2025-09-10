# controllers/ecg_controller.py
"""
ECGController —— BITalino 串流 → 濾波 → R 峰 → RR/HR/HRV（時域）
→ 繪圖（掃描式就緒版＋R峰紅點＋RR計數）

更新重點（2025-09）：
1) 紅點偵測 與 RR/HR 計數 完全分離：
   - 紅點：det_vis（沿用你的 ECGDetector）
   - RR/HR：det_rr（第二顆偵測器，容錯較寬）
2) 紅點每一幀都重新投影（不會隨時間漂移）。
3) 暫停/續傳不清空 RR；只有「開始」會清空。

可在 config.toml 另外加 [detector_rr] 覆寫 RR 偵測參數；沒寫有內建預設。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QLabel, QMessageBox, QFileDialog

from bitalino_helpers import BitalinoClient

# ─────────────────────────────────────────────────────────────
# 濾波：優先 ECGFilterRT；沒有就用 fallback
# ─────────────────────────────────────────────────────────────
_HAS_ECGFILTER = False
try:
    from processing.filters import ECGFilterRT, EnhancedBandpass
    _HAS_ECGFILTER = True
except Exception:
    try:
        import scipy.signal as _sig
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    class _OnePoleLPF:
        def __init__(self, fs: int, fc: float):
            a = float(np.exp(-2.0 * np.pi * fc / fs))
            self.a = a
            self.s = 0.0

        def filt_vec(self, x: np.ndarray) -> np.ndarray:
            y = np.empty_like(x, dtype=float)
            s = self.s
            a = self.a
            xv = x.astype(float)
            for i, v in enumerate(xv):
                s = a*s + (1-a)*v
                y[i] = s
            self.s = s
            return y

    class EnhancedBandpass:
        def __init__(self, fs: int, hp=0.67, lp=30.0, notch_freq: float = 0.0):
            self.fs = int(fs)
            self.hp_lp = _OnePoleLPF(fs, hp)
            self.lp_lp = _OnePoleLPF(fs, lp)
            self.use_notch = False
            if notch_freq and _HAS_SCIPY:
                b, a = _sig.iirnotch(w0=notch_freq/(fs/2.0), Q=30.0)
                self.sos_notch = _sig.tf2sos(b, a)
                self.zi_notch = _sig.sosfilt_zi(self.sos_notch)
                self.use_notch = True

        def process(self, x: np.ndarray) -> np.ndarray:
            base = self.hp_lp.filt_vec(x)
            y = x - base
            if self.use_notch:
                y, self.zi_notch = _sig.sosfilt(self.sos_notch, y, zi=self.zi_notch)
            y = self.lp_lp.filt_vec(y)
            if '_HAS_SCIPY' in globals() and _HAS_SCIPY and len(y) >= 5:
                y = _sig.medfilt(y, kernel_size=5)
            return y

        def set_notch_enabled(self, enabled: bool):
            self.use_notch = bool(enabled)

        def reset_state(self):
            self.hp_lp.s = 0.0
            self.lp_lp.s = 0.0


# ─────────────────────────────────────────────────────────────
# R 峰偵測器（沿用你的算法，不改）
# ─────────────────────────────────────────────────────────────
class ECGDetector:
    def __init__(
        self,
        fs: int,
        refractory_ms: float = 240.0,
        slope_k: float = 4.0,
        pre_lock_ms: float = 8.0,
        post_lock_ms: float = 70.0,
        tail_ms: float = 200.0,
        amp_abs_gate: float | None = None,
        k_mad_height: float = 2.8
    ):
        self.fs = int(fs)
        self.refractory = int(round(refractory_ms * fs / 1000.0))
        self.pre_lock = int(round(pre_lock_ms * fs / 1000.0))
        self.post_lock = int(round(post_lock_ms * fs / 1000.0))
        self.tail_len = int(round(tail_ms * fs / 1000.0))
        self.slope_k = float(slope_k)
        self.amp_abs_gate = None if amp_abs_gate is None else float(amp_abs_gate)
        self.k_mad_height = float(k_mad_height)

        self._y_tail = np.zeros(self.tail_len, dtype=float)
        self._global_i = 0
        self._last_peak_i = -10_000

        self.r_indices: List[int] = []
        self.rr_ms: List[float] = []

    def process(self, y_block: np.ndarray) -> None:
        if y_block.size == 0:
            return

        ycat = np.concatenate([self._y_tail, y_block])
        cat0_global = self._global_i - self.tail_len

        dy = np.diff(ycat, prepend=ycat[0])
        med = float(np.median(ycat))
        mad = float(np.median(np.abs(ycat - med))) + 1e-9
        slope_thr = self.slope_k * (float(np.median(np.abs(dy - np.median(dy)))) + 1e-9)
        height_thr = med + self.k_mad_height * mad
        if self.amp_abs_gate is not None:
            height_thr = max(height_thr, self.amp_abs_gate)

        self.r_indices.clear()
        self.rr_ms.clear()

        i = self.tail_len
        end = self.tail_len + len(y_block) - 1

        while i <= end:
            gi = cat0_global + i
            if gi - self._last_peak_i < self.refractory:
                i += 1
                continue

            if dy[i] > slope_thr and ycat[i] > height_thr:
                s = max(0, i - self.pre_lock)
                e = min(len(ycat) - 1, i + self.post_lock)
                seg = ycat[s:e+1]
                if seg.size == 0:
                    i += 1
                    continue

                c2 = s + int(np.argmax(seg))
                gi_peak = cat0_global + c2

                L = 4
                ok_shape = True
                if c2 - L >= 0 and c2 + L < len(ycat):
                    left = float(ycat[c2] - ycat[c2 - L])
                    right = float(ycat[c2 + L] - ycat[c2])
                    min_slope = 0.5 * mad
                    if not (left > min_slope and -right > min_slope):
                        ok_shape = False

                if ok_shape and ycat[c2] > height_thr:
                    self.r_indices.append(gi_peak)
                    if self._last_peak_i >= 0:
                        rr = (gi_peak - self._last_peak_i) * 1000.0 / self.fs
                        if 300.0 <= rr <= 2000.0:
                            self.rr_ms.append(rr)
                    self._last_peak_i = gi_peak
                    i = e + 1
                    continue

            i += 1

        self._global_i += len(y_block)
        self._y_tail = ycat[-self.tail_len:].copy()

    @property
    def global_index(self) -> int:
        return self._global_i


# ─────────────────────────────────────────────────────────────
# HRV（時域）
# ─────────────────────────────────────────────────────────────
@dataclass
class TimeDomainHRV:
    count: int
    mean_rr: float
    sdnn: float
    rmssd: float
    mean_hr: float


def compute_time_domain(rr_ms: List[float]) -> Optional[TimeDomainHRV]:
    if len(rr_ms) < 2:
        return None
    rr = np.asarray(rr_ms, dtype=float)
    mean_rr = float(rr.mean())
    sdnn = float(rr.std(ddof=1)) if rr.size > 1 else 0.0
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs*diffs))) if diffs.size > 0 else 0.0
    mean_hr = 60000.0/mean_rr if mean_rr > 0 else 0.0
    return TimeDomainHRV(len(rr_ms), mean_rr, sdnn, rmssd, mean_hr)


# ─────────────────────────────────────────────────────────────
# Qt 資料橋
# ─────────────────────────────────────────────────────────────
class DataBridge(QObject):
    arrived = pyqtSignal(object)  # numpy.ndarray


# ─────────────────────────────────────────────────────────────
# 主控制器
# ─────────────────────────────────────────────────────────────
class ECGController:
    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        lbl_rt_hr, lbl_stable_hr, status_bar,
        btn_save_rr, btn_analyze, cfg: Dict[str, Any],
        lbl_rr_count: Optional[QLabel] = None,
    ):
        # UI
        self.plot = plot_widget
        self.lbl_rt = lbl_rt_hr
        self.lbl_stable = lbl_stable_hr
        self.status_bar = status_bar
        self.btn_save_rr = btn_save_rr
        self.btn_analyze = btn_analyze
        self.lbl_rr_count = lbl_rr_count

        # 基本/裝置
        self.fs = int(cfg.get("sampling_rate", 1000))
        ch = cfg.get("analog_channels", [1])
        if not isinstance(ch, list):
            ch = [int(ch)]
        self.analog_channels: List[int] = ch
        self.address: Optional[str] = cfg.get("address") or None

        # 濾波
        f_cfg = cfg.get("filter", {})
        if f_cfg.get("enable", True) and _HAS_ECGFILTER:
            self.ecg_filter = ECGFilterRT(
                fs=self.fs,
                band=tuple(f_cfg.get("band", (0.5, 25.0))),
                notch=float(f_cfg.get("notch", 60.0)),
                order=int(f_cfg.get("order", 4)),
                q=float(f_cfg.get("q", 30.0)),
            )
            self._use_fallback = False
        elif f_cfg.get("enable", True):
            self.ecg_filter = EnhancedBandpass(
                self.fs, hp=0.67, lp=30.0, notch_freq=float(f_cfg.get("notch", 0.0)))
            self._use_fallback = True
        else:
            self.ecg_filter = None
            self._use_fallback = False
        self.filter_enabled = bool(f_cfg.get("enable", True))
        self.notch_enabled = bool(f_cfg.get("notch", 60.0) and not self._use_fallback)
        if self.ecg_filter is not None and hasattr(self.ecg_filter, "set_notch_enabled"):
            try:
                self.ecg_filter.set_notch_enabled(self.notch_enabled)
            except Exception:
                pass

        # 繪圖 / 緩衝
        plot_cfg = cfg.get("plot", {})
        self.seconds_window = int(plot_cfg.get("seconds", 10))
        self.buf_len = max(10, int(self.seconds_window*self.fs))
        self.gain = float(plot_cfg.get("gain", 1.5))
        self.ecg_col = int(plot_cfg.get("ecg_col", -1))
        self.chunk = int(plot_cfg.get("chunk", 100))

        # 顯示模式
        self.mode = str(plot_cfg.get("mode", "sweep")).strip().lower()
        self.direction = str(plot_cfg.get("direction", "ltr")).strip().lower()

        # ring buffer 與座標
        self._ybuf = np.zeros(self.buf_len, dtype=float)
        self._tbase = np.linspace(0.0, float(self.seconds_window), self.buf_len, endpoint=False)

        # 繪圖全域樣本計數（紅點對齊用）
        self._g_draw = 0

        # 掃描狀態
        self._pos = 0
        self._wrapped_once = False

        # 主曲線
        self.curve = self.plot.plot(self._tbase, self._ybuf, pen=pg.mkPen(width=2))
        self.curve.setDownsampling(auto=True)
        self.curve.setClipToView(True)

        # 紅點圖層
        self._rwin: Deque[int] = deque()
        self.scatter = pg.ScatterPlotItem(size=8, pen=None, brush='r')
        self.plot.addItem(self.scatter)

        # 座標軸/外觀
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.showGrid(x=True, y=True)
        self.plot.getViewBox().setXRange(0, self.seconds_window, padding=0)
        vb = self.plot.getViewBox()
        vb.setDefaultPadding(float(plot_cfg.get("y_padding", 0.20)))
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.invertX(False)

        # 偵測/顯示狀態
        self._det_cfg = cfg.get("detector", {})
        self._det_rr_cfg = cfg.get("detector_rr", {})  # ← 新增：RR 專用參數（可不設）
        
        # 確保檢測器被正確初始化
        self.det_vis = None
        self.det_rr = None
        self._initialize_detectors()

        self.warmup_left = int(0.4 * self.fs)
        self.alpha = 0.12
        self.hr_stable = None
        self._rr_accum: List[float] = []

        # 視覺紅點延遲補償（只影響顯示）
        default_vis_ms = float(self._det_cfg.get("vis_lag_ms", 0.0))
        self.rpeak_vis_lag_ms = default_vis_ms
        self._rpeak_vis_lag_samples = int(round(self.rpeak_vis_lag_ms * self.fs / 1000.0))

        # 橋接/Client/UI
        self.bridge = DataBridge()
        self.bridge.arrived.connect(self._on_arrived_mainthread)
        self.client = BitalinoClient()
        self.client.configure(address=self.address, sampling_rate=self.fs, analog_channels=self.analog_channels)
        self.btn_save_rr.clicked.connect(self._on_save_rr_clicked)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        self._set_hr("--", "--")
        self._set_rr_count(0)

        # 電池
        b_cfg = cfg.get("battery", {})
        self.batt_enabled = bool(b_cfg.get("enable", True))
        self.batt_poll_ms = int(b_cfg.get("poll_s", 15))*1000
        self.batt_raw_min = float(b_cfg.get("raw_min", 511))
        self.batt_raw_max = float(b_cfg.get("raw_max", 645))
        self.batt_low_pct = float(b_cfg.get("low_pct", 20))
        self.batt_crit_pct = float(b_cfg.get("critical_pct", 10))
        self.batt_set_dev_pct = int(b_cfg.get("set_device_threshold_pct", 10))
        self.batt_label = QLabel("🔋 --%")
        if self.batt_enabled:
            self.status_bar.addPermanentWidget(self.batt_label)
        self._batt_timer = QTimer(self.bridge)
        self._batt_timer.setInterval(self.batt_poll_ms)
        self._batt_timer.timeout.connect(self._on_batt_tick)

        # 吞吐監看
        self.rx_label = QLabel("📡 0/s")
        self.status_bar.addPermanentWidget(self.rx_label)
        self._rx_counter = 0
        self._rx_last = 0
        self._rx_zero_ticks = 0
        self._rx_timer = QTimer(self.bridge)
        self._rx_timer.setInterval(1000)
        self._rx_timer.timeout.connect(self._on_rx_tick)

        self._is_streaming = False
        self._is_paused = False

    def _initialize_detectors(self):
        """確保檢測器被正確初始化"""
        self.det_vis = self._make_detector_vis()
        self.det_rr = self._make_detector_rr()

    # ── 偵測器建構 ─────────────────────────────────────────────
    def _make_detector_vis(self) -> ECGDetector:
        d = self._det_cfg
        detector = ECGDetector(
            fs=self.fs,
            refractory_ms=float(d.get("refractory_ms", 280.0)),
            slope_k=float(d.get("slope_k", 3.5)),
            pre_lock_ms=float(d.get("pre_lock_ms", 10.0)),
            post_lock_ms=float(d.get("post_lock_ms", 80.0)),
            tail_ms=float(d.get("tail_ms", 250.0)),
            amp_abs_gate=float(d.get("amp_abs_gate", 180.0)),
            k_mad_height=float(d.get("k_mad_height", 2.5)),
        )
        print(f"視覺檢測器初始化成功: {detector is not None}")
        return detector

    def _make_detector_rr(self) -> ECGDetector:
        """
        RR 專用偵測器：專門用於RR計算，參數更寬鬆
        """
        s = self._det_rr_cfg or {}
        if s:
            # 如果配置中有專門的RR檢測器參數，使用它們
            detector = ECGDetector(
                fs=self.fs,
                refractory_ms=float(s.get("refractory_ms", 250.0)),
                slope_k=float(s.get("slope_k", 2.8)),
                pre_lock_ms=float(s.get("pre_lock_ms", 15.0)),
                post_lock_ms=float(s.get("post_lock_ms", 100.0)),
                tail_ms=float(s.get("tail_ms", 300.0)),
                amp_abs_gate=float(s.get("amp_abs_gate", 100.0)),
                k_mad_height=float(s.get("k_mad_height", 2.0)),
            )
        else:
            # 使用預設的寬鬆參數
            detector = ECGDetector(
                fs=self.fs,
                refractory_ms=250.0,
                slope_k=2.8,
                pre_lock_ms=15.0,
                post_lock_ms=100.0,
                tail_ms=300.0,
                amp_abs_gate=100.0,
                k_mad_height=2.0,
            )
        
        print(f"RR檢測器初始化成功: {detector is not None}")
        return detector

    # ── 濾波/Notch ─────────────────────────────────────────────
    def set_filter_enabled(self, enabled: bool):
        self.filter_enabled = bool(enabled)
        self.status_bar.showMessage(f"濾波：{'開' if self.filter_enabled else '關'}", 1500)

    def set_notch_enabled(self, enabled: bool):
        self.notch_enabled = bool(enabled)
        if self.ecg_filter is not None and hasattr(self.ecg_filter, "set_notch_enabled"):
            try:
                self.ecg_filter.set_notch_enabled(self.notch_enabled)
                self.status_bar.showMessage(f"Notch 60Hz：{'開' if self.notch_enabled else '關'}", 1500)
            except Exception:
                self.status_bar.showMessage("Notch 切換失敗（濾波器不支援）", 3000)
        else:
            self.status_bar.showMessage("目前濾波器不支援 Notch（未安裝 scipy 或 notch=0）", 3000)

    # 方向/座標
    def _apply_direction(self):
        try:
            vb = self.plot.getPlotItem().vb
            if self.mode == "sweep":
                vb.invertX(False)
            else:
                vb.invertX(self.direction == "rtl")
            vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            vb.setXRange(0, self.seconds_window, padding=0)
        except Exception:
            pass

    # 連線/斷線
    def connect_device(self, address: Optional[str] = None, retries: int = 3) -> bool:
        try:
            self.status_bar.showMessage("連線中…")
            if address:
                self.address = address
            try:
                self.client.configure(
                    address=self.address,
                    sampling_rate=self.fs,
                    analog_channels=self.analog_channels
                )
            except Exception:
                pass

            self.client.connect(retries=retries)
            try:
                self._set_batt_label(self._query_battery_percent())
                if self.batt_enabled and getattr(self.client, "device", None):
                    self.client.device.battery(int(self.batt_set_dev_pct))
            except Exception:
                pass
            self._start_battery_monitor()
            self.status_bar.showMessage("BITalino 連線成功", 2500)
            return True
        except Exception as e:
            self.status_bar.showMessage(f"連線失敗：{e}", 6000)
            return False

    def disconnect_device(self):
        try:
            self.stop_stream()
            self.client.close()
            self._stop_battery_monitor()
            self.status_bar.showMessage("已斷線", 2500)
        except Exception as e:
            self.status_bar.showMessage(f"斷線錯誤：{e}", 5000)

    # Callbacks
    def _assign_callbacks(self):
        def _on_data(arr):
            try:
                self.bridge.arrived.emit(np.asarray(arr))
            except Exception as ex:
                print("UI dispatch error:", ex)
        self.client.data_callback = _on_data

        def _on_err(e):
            self._is_streaming = False
            self._is_paused = True
            try:
                self.client.stop_acquisition()
            except Exception:
                pass
            self._stop_battery_monitor()
            self._rx_timer.stop()
            self.status_bar.showMessage(f"資料擷取中斷：{e}", 8000)
        setattr(self.client, "on_error", _on_err)

    # 開始（全新）
    def start_stream(self) -> bool:
        if getattr(self, "_is_streaming", False):
            self.status_bar.showMessage("已在擷取中", 1500)
            return True
        if not hasattr(self, "_is_paused"):
            self._is_paused = False
        try:
            try:
                if getattr(self.client, "is_acquiring", False):
                    self.client.stop_acquisition()
            except Exception:
                pass

            self._assign_callbacks()

            if not getattr(self.client, "is_connected", False):
                self.client.connect(retries=3)

            # 復位濾波與兩顆偵測器
            self.warmup_left = int(0.4 * self.fs)
            if getattr(self, "ecg_filter", None) is not None and hasattr(self.ecg_filter, "reset_state"):
                try:
                    self.ecg_filter.reset_state()
                except Exception:
                    pass
            
            # 確保檢測器被初始化
            if self.det_vis is None:
                self.det_vis = self._make_detector_vis()
            if self.det_rr is None:
                self.det_rr = self._make_detector_rr()

            # 清狀態（只有開始會清空）
            self._rr_accum.clear()
            self.hr_stable = None
            self._set_hr("--", "--")
            self._set_rr_count(0)

            self._ybuf[:] = 0.0
            self.curve.setData(self._tbase, self._ybuf)
            self._pos = 0
            self._wrapped_once = False
            self._rwin.clear()
            self.scatter.setData([], [])
            self._g_draw = 0
            self._apply_direction()

            chunk = max(10, min(self.buf_len // 2, int(self.chunk)))
            self.client.start_acquisition(chunk_size=chunk)
            self._is_streaming = True
            self._is_paused = False

            self._rx_counter = 0
            self._rx_last = 0
            self._rx_zero_ticks = 0
            self._rx_timer.start()
            self._start_battery_monitor()

            self.status_bar.showMessage(
                f"擷取中：fs={self.fs}, ch={self.analog_channels}, chunk={chunk}, mode={self.mode}", 3500
            )
            return True

        except Exception as e:
            self._is_streaming = False
            try:
                self.client.stop_acquisition()
            except Exception:
                pass
            self._rx_timer.stop()
            try:
                self.client.data_callback = None
            except Exception:
                pass
            self.status_bar.showMessage(f"開始擷取失敗：{e}", 6000)
            try:
                QMessageBox.critical(None, "開始擷取失敗", str(e))
            except Exception:
                pass
            return False

    # 停止
    def stop_stream(self):
        try:
            self._is_streaming = False
            self._rx_timer.stop()
            try:
                self.client.stop_acquisition()
            except Exception:
                pass
            try:
                self.client.data_callback = None
            except Exception:
                pass
            self._on_batt_tick()
            self.status_bar.showMessage("已停止擷取", 1800)
        except Exception as e:
            self.status_bar.showMessage(f"停止擷取失敗：{e}", 4000)

    # 暫停 / 續傳
    def pause_stream(self):
        if not self._is_streaming:
            self.status_bar.showMessage("目前未在擷取中", 1500)
            return
        try:
            self._is_streaming = False
            self._is_paused = True
            self._rx_timer.stop()
            try:
                self.client.stop_acquisition()
            except Exception:
                pass
            self.status_bar.showMessage("已暫停（保留緩衝與偵測狀態，可續傳）", 2500)
        except Exception as e:
            self.status_bar.showMessage(f"暫停失敗：{e}", 4000)

    def resume_stream(self) -> bool:
        if not getattr(self.client, "is_connected", False):
            self.status_bar.showMessage("尚未連線，無法續傳。請先連線。", 2500)
            return False
        if not self._is_paused:
            return self.start_stream()
        try:
            try:
                if getattr(self.client, "is_acquiring", False):
                    self.client.stop_acquisition()
            except Exception:
                pass

            self._assign_callbacks()

            chunk = max(10, min(self.buf_len//2, int(self.chunk)))
            self.client.start_acquisition(chunk_size=chunk)

            self._rx_counter = 0
            self._rx_last = 0
            self._rx_zero_ticks = 0
            self._rx_timer.start()
            self._start_battery_monitor()

            self._is_streaming = True
            self._is_paused = False
            self.status_bar.showMessage("續傳中（狀態已保留）", 2000)
            return True
        except Exception as e:
            self.status_bar.showMessage(f"續傳失敗：{e}", 5000)
            return False

    # 主執行緒：接資料 → 濾波/偵測/繪圖
    def _on_arrived_mainthread(self, arr_obj: object):
        self._apply_direction()

        data = np.atleast_2d(np.asarray(arr_obj))
        if data.shape[0] == 0:
            return

        self._rx_counter += data.shape[0]
        if not hasattr(self, "_first_chunk_seen"):
            self._first_chunk_seen = True
            self.status_bar.showMessage(f"✓ 已收到資料：shape={data.shape}", 2500)

        ecg = data[:, -1].astype(float) if self.ecg_col == -1 else data[:, int(self.ecg_col)].astype(float)

        # 濾波 + 增益
        if self.filter_enabled and self.ecg_filter is not None:
            try:
                y = self.ecg_filter.process(ecg) * self.gain
            except Exception:
                y = ecg * self.gain
        else:
            y = ecg * self.gain

        # 畫圖（更新 _g_draw）
        if self.mode == "sweep":
            self._plot_sweep(y)
        else:
            self._plot_sliding(y)

        # 暖機
        n = len(y)
        if self.warmup_left > 0:
            self.warmup_left -= n
            self._prune_rwin_and_update_scatter()
            return
        
        # 確保檢測器存在
        if self.det_rr is None:
            self.det_rr = self._make_detector_rr()
        if self.det_vis is None:
            self.det_vis = self._make_detector_vis()
        
        # 1. 先独立计算RR间隔（优先级最高）
        self._update_rr_calculation(y)
        
        # 2. 然后进行R-Peak可视化（次要）
        self._update_rpeak_visualization(y)

        # ★ 每一幀都重算紅點投影（不再漂移）
        self._prune_rwin_and_update_scatter()

    # 掃描式
    def _plot_sweep(self, y: np.ndarray):
        L = self._ybuf.size
        n = int(y.size)
        if n <= 0 or L == 0:
            return

        pos = self._pos
        if n >= L:
            self._ybuf[:] = y[-L:]
            pos = 0
            self._wrapped_once = True
        else:
            end = pos + n
            if end <= L:
                self._ybuf[pos:end] = y
                pos = end
                if pos == L:
                    pos = 0
                    self._wrapped_once = True
            else:
                k = L - pos
                self._ybuf[pos:] = y[:k]
                self._ybuf[:n-k] = y[k:]
                pos = (n - k)
                self._wrapped_once = True

        self._pos = pos

        y_vis = self._ybuf.copy()
        if not self._wrapped_once:
            if self._pos < L:
                y_vis[self._pos:] = np.nan
        else:
            y_vis[self._pos % L] = np.nan

        self.curve.setData(self._tbase, y_vis, connect='finite')
        self._g_draw += n

    # 滑動窗
    def _plot_sliding(self, y: np.ndarray):
        L = self._ybuf.size
        n = int(y.size)
        if n <= 0:
            return
        if n >= L:
            self._ybuf[:] = y[-L:]
        else:
            self._ybuf = np.roll(self._ybuf, -n)
            self._ybuf[-n:] = y
        self.curve.setData(self._tbase, self._ybuf)
        self._g_draw += n

    def _update_rr_calculation(self, y: np.ndarray):
        """
        独立计算RR间隔，不依赖视觉检测器
        使用更稳定的事后处理算法
        """
        if len(y) == 0 or self.det_rr is None:
            return
        
        try:
            # 使用专门的RR检测器处理数据
            self.det_rr.process(y)
            
            if not self.det_rr.rr_ms:
                return
            
            # 获取新检测到的RR间隔
            new_rr_intervals = self.det_rr.rr_ms.copy()
            
            # 质量检查：移除异常值
            filtered_rr = self._validate_rr_intervals(new_rr_intervals)
            
            if not filtered_rr:
                return
            
            # 添加到累积列表
            self._rr_accum.extend(filtered_rr)
            
            # 计算实时心率
            instant_hr, stable_hr = self._calculate_heart_rate(filtered_rr)
            
            # 更新UI
            self._set_hr(f"{instant_hr:.0f}", f"{stable_hr:.0f}")
            self._set_rr_count(len(self._rr_accum))
            
        except AttributeError as e:
            print(f"RR检测器错误: {e}")
            # 尝试重新初始化
            self.det_rr = self._make_detector_rr()

    def _update_rpeak_visualization(self, y: np.ndarray):
        """
        只负责R-Peak的可视化，不参与RR计算
        """
        if self.det_vis is None:
            return
        
        try:
            # 使用视觉检测器
            self.det_vis.process(y)
            
            if self.det_vis.r_indices:
                for gi in self.det_vis.r_indices:
                    self._rwin.append(gi)
            
        except AttributeError as e:
            print(f"视觉检测器错误: {e}")
            # 尝试重新初始化
            self.det_vis = self._make_detector_vis()

    def _validate_rr_intervals(self, rr_intervals: List[float]) -> List[float]:
        """
        RR间隔质量检查和校正
        """
        if len(rr_intervals) < 1:
            return rr_intervals
        
        valid_rr = []
        for i, rr in enumerate(rr_intervals):
            # 基本范围检查 (300-1200ms = 50-200 bpm)
            if not (300 <= rr <= 1200):
                continue
            
            # 相对于前一个RR的变化率检查
            if i > 0 and valid_rr:
                prev_rr = valid_rr[-1]
                change_ratio = abs(rr - prev_rr) / prev_rr
                if change_ratio > 0.3:  # 变化超过30%视为异常
                    continue
            
            valid_rr.append(rr)
        
        return valid_rr

    def _calculate_heart_rate(self, rr_intervals: List[float]) -> Tuple[float, float]:
        """
        更稳健的心率计算
        """
        if not rr_intervals:
            return 0.0, 0.0
        
        # 使用中位数而不是平均值，对异常值更稳健
        recent_rr = rr_intervals[-min(5, len(rr_intervals)):]
        median_rr = np.median(recent_rr)
        instant_hr = 60000.0 / median_rr
        
        # 平滑处理
        if self.hr_stable is None:
            stable_hr = instant_hr
        else:
            # 使用自适应平滑系数
            alpha = 0.2 if abs(instant_hr - self.hr_stable) > 10 else 0.1
            stable_hr = (1 - alpha) * self.hr_stable + alpha * instant_hr
        
        self.hr_stable = stable_hr
        return instant_hr, stable_hr

    def _prune_rwin_and_update_scatter(self):
        if not self._rwin:
            self.scatter.setData([], [])
            return

        L = self.buf_len
        if L <= 0:
            self.scatter.setData([], [])
            return

        # 计算当前显示的全局索引范围
        if self.mode == "sweep":
            # 扫描模式：当前显示的是从 (g_now - L) 到 g_now 的数据
            g_now = self._g_draw
            g_min = max(0, g_now - L)
        else:
            # 滑动窗口模式：总是显示最新的 L 个样本
            g_now = self.det_vis.global_index if self.det_vis else self._g_draw
            g_min = max(0, g_now - L)

        # 移除窗口外的点
        while self._rwin and self._rwin[0] < g_min:
            self._rwin.popleft()

        if not self._rwin:
            self.scatter.setData([], [])
            return

        xs: List[float] = []
        ys: List[float] = []
        
        for gi in self._rwin:
            # 计算在缓冲区中的位置
            if self.mode == "sweep":
                # 扫描模式：计算相对位置
                rel_pos = gi - g_min
                if 0 <= rel_pos < L:
                    xs.append(float(self._tbase[rel_pos]))
                    ys.append(float(self._ybuf[rel_pos]))
            else:
                # 滑动窗口模式：使用模运算
                idx = gi % L
                xs.append(float(self._tbase[idx]))
                ys.append(float(self._ybuf[idx]))
        
        self.scatter.setData(xs, ys)

    # RR 存檔 / HRV（時域）
    def _on_save_rr_clicked(self):
        if len(self._rr_accum) == 0:
            QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn, _ = QFileDialog.getSaveFileName(None, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)")
        if not fn:
            return
        Path(fn).write_text("\n".join(f"{v:.1f}" for v in self._rr_accum), encoding="utf-8")
        QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    def _on_analyze_clicked(self):
        res = compute_time_domain(self._rr_accum)
        if res is None:
            QMessageBox.information(None, "HRV 分析", "RR 數量不足，請先擷取 RR。")
            return
        msg = (f"RR 數量：{res.count}\n"
               f"Mean RR：{res.mean_rr:.1f} ms\n"
               f"SDNN：{res.sdnn:.1f} ms\n"
               f"RMSSD：{res.rmssd:.1f} ms\n"
               f"Mean HR：{res.mean_hr:.1f} bpm\n")
        QMessageBox.information(None, "HRV（時域）", msg)

    # UI 小工具
    def _set_hr(self, rt: str, stable: str):
        try:
            self.lbl_rt.setText(f"即時心跳：{rt} bpm")
            self.lbl_stable.setText(f"穩定心跳：{stable} bpm")
        except Exception:
            pass

    def _set_rr_count(self, n: int):
        try:
            if self.lbl_rr_count is not None:
                self.lbl_rr_count.setText(f"RR 計數：{n}")
        except Exception:
            pass

    # 吞吐監看
    def _on_rx_tick(self):
        now = self._rx_counter
        rate = max(0, now - self._rx_last)
        self._rx_last = now
        self.rx_label.setText(f"📡 {rate}/s")
        if self._is_streaming:
            if rate == 0:
                self._rx_zero_ticks += 1
                if self._rx_zero_ticks >= 2:
                    self.status_bar.showMessage("⚠ 未收到資料：請確認 COM/電源/取樣與接線", 4000)
            else:
                self._rx_zero_ticks = 0

    # 電池
    def _battery_percent_from_raw(self, raw: float) -> float:
        r0, r1 = self.batt_raw_min, self.batt_raw_max
        if r1 <= r0:
            return 0.0
        pct = 1.0 + (float(raw) - r0) * (98.0 / (r1 - r0))
        return max(0.0, min(100.0, pct))

    def _query_battery_percent(self):
        dev = getattr(self.client, "device", None)
        if dev is None:
            return None
        try:
            st = dev.state()
            raw = st.get("battery") if isinstance(st, dict) else None
            if raw is None:
                return None
            return self._battery_percent_from_raw(raw)
        except Exception:
            return None

    def _set_batt_label(self, pct: Optional[float]):
        if not self.batt_enabled:
            return
        if pct is None:
            self.batt_label.setText("🔋 --%")
            self.batt_label.setStyleSheet("")
            return
        self.batt_label.setText(f"🔋 {pct:0.0f}%")
        if pct <= self.batt_crit_pct:
            self.batt_label.setStyleSheet("color:#e53935;")
        elif pct <= self.batt_low_pct:
            self.batt_label.setStyleSheet("color:#fb8c00;")
        else:
            self.batt_label.setStyleSheet("")

    def _on_batt_tick(self):
        if getattr(self.client, "is_acquiring", False):
            return
        pct = self._query_battery_percent()
        self._set_batt_label(pct)
        if pct is not None and pct <= self.batt_crit_pct:
            self.status_bar.showMessage("電量過低：請儘快充電", 2500)

    def _start_battery_monitor(self):
        if self.batt_enabled:
            self._batt_timer.start()
            self._on_batt_tick()

    def _stop_battery_monitor(self):
        try:
            self._batt_timer.stop()
            self._set_batt_label(None)
        except Exception:
            pass