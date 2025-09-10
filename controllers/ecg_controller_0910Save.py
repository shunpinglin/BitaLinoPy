# controllers/ecg_controller.py
"""
ECGController —— BITalino 串流 → 濾波 → R 峰 → RR/HR/HRV（時域）
→ 繪圖（掃描式＋R 峰紅點）

重點：
1) RR 計數改用 RPeakTracker（自適應門檻 + 搜尋回補 + 拋物線次取樣），
   與 UI 紅點完全解耦，專注精準 RR 以供 FFT/MSE。
2) 紅點只做視覺，不影響 RR；採 ring-buffer 直接映射，長時間不飄移。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Deque, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import time, math
from datetime import datetime

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
            a = float(math.exp(-2.0 * math.pi * fc / fs))
            self.a = a
            self.s = 0.0

        def filt_vec(self, x: np.ndarray) -> np.ndarray:
            y = np.empty_like(x, dtype=float)
            s = self.s; a = self.a
            xv = x.astype(float)
            for i, v in enumerate(xv):
                s = a * s + (1 - a) * v
                y[i] = s
            self.s = s
            return y

    class EnhancedBandpass:
        """簡化帶通（高通基線扣除 + 低通），選配 Notch（若裝了 scipy）。"""
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
# 供參考：你原本演算法（保留，但不再用於 RR）
# ─────────────────────────────────────────────────────────────
class ECGDetector:
    def __init__(
        self, fs: int, refractory_ms: float = 240.0, slope_k: float = 4.0,
        pre_lock_ms: float = 8.0, post_lock_ms: float = 70.0,
        tail_ms: float = 200.0, amp_abs_gate: float | None = None,
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
        self.r_indices.clear(); self.rr_ms.clear()

        i = self.tail_len; end = self.tail_len + len(y_block) - 1
        while i <= end:
            gi = cat0_global + i
            if gi - self._last_peak_i < self.refractory:
                i += 1; continue
            if dy[i] > slope_thr and ycat[i] > height_thr:
                s = max(0, i - self.pre_lock)
                e = min(len(ycat) - 1, i + self.post_lock)
                seg = ycat[s:e+1]
                if seg.size == 0:
                    i += 1; continue
                c2 = s + int(np.argmax(seg))
                gi_peak = cat0_global + c2
                L = 4; ok_shape = True
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
                    i = e + 1; continue
            i += 1

        self._global_i += len(y_block)
        self._y_tail = ycat[-self.tail_len:].copy()

    @property
    def global_index(self) -> int:
        return self._global_i


# ─────────────────────────────────────────────────────────────
# 新：高精度即時 R-peak 追蹤（用於 RR/HR）
# ─────────────────────────────────────────────────────────────
class RPeakTracker:
    """
    以已濾波訊號為輸入（y），線上計算：
      - d/dt -> square -> 指數移動平均(≈120ms) 形成能量包絡
      - 自適應門檻（Spki/Npki） + 不應期
      - 搜尋回補：超過 1.66×平均 RR 尚未偵測則強制在區段最高能量處搜尋
      - 在包絡峰附近到 y 上鎖 apex，並以拋物線次取樣修正時間戳
    產出：
      - r_idx_display: 供畫點的整數全域 index
      - rr_ms: 以「次取樣精度」時間戳計得的 RR（ms）
    """
    def __init__(self, fs: int, env_ms: float = 120.0, search_ms: float = 80.0, refractory_ms: float = 240.0):
        self.fs = int(fs)
        tau = float(env_ms) / 1000.0
        self.alpha = math.exp(-1.0 / (self.fs * tau))  # EMA 係數
        self.s_env = 0.0
        self._env_tail = np.zeros(2, dtype=float)      # 峰值偵測用
        self._bp_tail = np.zeros(int(round(search_ms * fs / 1000.0)) + 3, dtype=float)
        self.search = int(round(search_ms * fs / 1000.0))
        self.refractory = int(round(refractory_ms * fs / 1000.0))

        self.Spki = 0.0
        self.Npki = 0.0
        self._th = 0.0

        self.global_i = 0
        self._last_r_pos: Optional[float] = None
        self._last_accept_i: int = -10_000
        self._rr_avg: Optional[float] = None  # ms

    def reset(self):
        self.s_env = 0.0
        self._env_tail[:] = 0.0
        self._bp_tail[:] = 0.0
        self.Spki = 0.0; self.Npki = 0.0; self._th = 0.0
        self.global_i = 0
        self._last_r_pos = None
        self._last_accept_i = -10_000
        self._rr_avg = None

    def _ema_env(self, y: np.ndarray) -> np.ndarray:
        # 以微分平方做能量，EMA 平滑（對齊樣本，不引入群延遲）
        d = np.diff(y, prepend=y[0])
        x = d * d
        out = np.empty_like(x, dtype=float)
        s = self.s_env; a = self.alpha
        for i, v in enumerate(x):
            s = a * s + (1 - a) * v
            out[i] = s
        self.s_env = s
        return out

    def process(self, y_block: np.ndarray) -> Tuple[List[int], List[float]]:
        if y_block.size == 0:
            return [], []

        start_g = self.global_i
        env = self._ema_env(y_block)

        # 峰值偵測（跨邊界保護）
        env_cat = np.concatenate([self._env_tail, env])
        pk_mask = (env_cat[1:-1] > env_cat[:-2]) & (env_cat[1:-1] >= env_cat[2:])
        pk_idx_cat = np.where(pk_mask)[0] + 1  # 相對於 env_cat
        bp_cat = np.concatenate([self._bp_tail, y_block])

        r_disp: List[int] = []
        rr_ms: List[float] = []

        for pk in pk_idx_cat:
            amp = float(env_cat[pk])
            # 動態門檻
            self._th = self.Npki + 0.25 * (self.Spki - self.Npki)

            # 轉成本區塊索引 / 全域索引
            i_block = pk - len(self._env_tail)
            if i_block < 0:
                continue
            gi_cand = start_g + i_block

            # 不應期
            if gi_cand - self._last_accept_i < self.refractory:
                self.Npki = 0.875 * self.Npki + 0.125 * amp
                continue

            # 判斷是否為有效峰
            if amp >= self._th:
                # 到原波形上鎖 apex（±search）
                center = len(self._bp_tail) + i_block
                pre = self.search; post = self.search
                sidx = max(0, center - pre)
                eidx = min(len(bp_cat) - 1, center + post)
                seg = bp_cat[sidx:eidx + 1]
                if seg.size == 0:
                    continue
                local = int(np.argmax(seg)) + sidx

                # 拋物線次取樣
                delta = 0.0
                if 1 <= local < len(bp_cat) - 1:
                    y1, y2, y3 = bp_cat[local - 1], bp_cat[local], bp_cat[local + 1]
                    denom = (y1 - 2 * y2 + y3)
                    if abs(denom) > 1e-12:
                        delta = 0.5 * (y1 - y3) / denom

                gi_peak = start_g + (local - len(self._bp_tail))
                pos_float = float(gi_peak) + float(delta)

                # RR（以次取樣精度）
                if self._last_r_pos is not None:
                    rr = (pos_float - self._last_r_pos) * 1000.0 / self.fs
                    if 300.0 <= rr <= 2000.0:
                        rr_ms.append(rr)
                        if self._rr_avg is None:
                            self._rr_avg = rr
                        else:
                            self._rr_avg = 0.9 * self._rr_avg + 0.1 * rr

                self._last_r_pos = pos_float
                self._last_accept_i = gi_cand
                r_disp.append(int(round(pos_float)))
                self.Spki = 0.875 * self.Spki + 0.125 * amp
            else:
                self.Npki = 0.875 * self.Npki + 0.125 * amp

        # 搜尋回補：若遠超平均 RR 仍未出現峰，強制在最高能量點找一次
        if self._rr_avg is not None and self._last_r_pos is not None:
            expected = 1.66 * (self._rr_avg * self.fs / 1000.0)  # 標準建議係數
            elapsed = (start_g + len(y_block)) - int(round(self._last_r_pos))
            if elapsed > expected:
                pk2 = int(np.argmax(env)) + len(self._env_tail)
                center = len(self._bp_tail) + (pk2 - len(self._env_tail))
                sidx = max(0, center - self.search)
                eidx = min(len(bp_cat) - 1, center + self.search)
                seg = bp_cat[sidx:eidx + 1]
                local = int(np.argmax(seg)) + sidx
                delta = 0.0
                if 1 <= local < len(bp_cat) - 1:
                    y1, y2, y3 = bp_cat[local - 1], bp_cat[local], bp_cat[local + 1]
                    denom = (y1 - 2 * y2 + y3)
                    if abs(denom) > 1e-12:
                        delta = 0.5 * (y1 - y3) / denom
                gi_peak = start_g + (local - len(self._bp_tail))
                pos_float = float(gi_peak) + float(delta)
                rr = (pos_float - self._last_r_pos) * 1000.0 / self.fs
                if 300.0 <= rr <= 2000.0:
                    rr_ms.append(rr)
                    self._rr_avg = 0.9 * self._rr_avg + 0.1 * rr if self._rr_avg is not None else rr
                    self._last_r_pos = pos_float
                    self._last_accept_i = int(pos_float)
                    r_disp.append(int(round(pos_float)))
                    self.Spki = 0.875 * self.Spki + 0.125 * float(np.max(env))

        # 更新尾巴與全域計數
        self._env_tail = env_cat[-2:].copy()
        tail_len = min(len(bp_cat), self.search + 3)
        self._bp_tail = bp_cat[-tail_len:].copy()
        self.global_i += len(y_block)
        return r_disp, rr_ms

    @property
    def global_index(self) -> int:
        return self.global_i


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
    rmssd = float(np.sqrt(np.mean(diffs * diffs))) if diffs.size > 0 else 0.0
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
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

        # 濾波器
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
        self.buf_len = max(10, int(self.seconds_window * self.fs))
        self.gain = float(plot_cfg.get("gain", 1.5))
        self.ecg_col = int(plot_cfg.get("ecg_col", -1))
        self.chunk = int(plot_cfg.get("chunk", 100))

        self.mode = str(plot_cfg.get("mode", "sweep")).strip().lower()
        self.direction = str(plot_cfg.get("direction", "ltr")).strip().lower()

        self._ybuf = np.zeros(self.buf_len, dtype=float)
        self._tbase = np.linspace(0.0, float(self.seconds_window), self.buf_len, endpoint=False)
        
        self._g_draw = 0                  # 自開始以來，已畫的樣本總數
        self._rr_anchor = 0               # RR 引擎索引 0 對應到畫面上的全域樣本序
        self._rr_started = False          # 是否已把第一包資料送進 RR 引擎

        self._pos = 0
        self._wrapped_once = False

        self.curve = self.plot.plot(self._tbase, self._ybuf, pen=pg.mkPen(width=2))
        self.curve.setDownsampling(auto=True)
        self.curve.setClipToView(True)

        # R 紅點圖層（僅視覺）
        self._rwin: Deque[int] = deque()
        self.scatter = pg.ScatterPlotItem(size=8, pen=None, brush='r')
        self.plot.addItem(self.scatter)

        # 坐標/外觀
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.showGrid(x=True, y=True)
        self.plot.getViewBox().setXRange(0, self.seconds_window, padding=0)
        y_pad = float(plot_cfg.get("y_padding", 0.20))
        vb = self.plot.getViewBox()
        vb.setDefaultPadding(y_pad)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.invertX(False)

        # 偵測/顯示狀態
        det_cfg = cfg.get("detector", {})
        self._det_cfg = det_cfg
        self.warmup_left = int(0.4 * self.fs)
        self.alpha = 0.12
        self.hr_stable = None

        # ★ 專用 RR 引擎（與紅點解耦）
        self.rr_engine = RPeakTracker(
            fs=self.fs,
            env_ms=float(det_cfg.get("env_ms", 120.0)),
            search_ms=float(det_cfg.get("search_ms", 80.0)),
            refractory_ms=float(det_cfg.get("refractory_ms", 240.0))
        )

        # 視覺 R-peak 延遲（僅影響紅點）
        default_vis_ms = float(det_cfg.get("vis_lag_ms", 0.0))
        self.rpeak_vis_lag_ms = default_vis_ms
        self._rpeak_vis_lag_samples = int(round(self.rpeak_vis_lag_ms * self.fs / 1000.0))

        # RR 累積（供 HRV/FFT/MSE）
        self._rr_accum: List[float] = []

        # 橋接/Client/UI
        self.bridge = DataBridge()
        self.bridge.arrived.connect(self._on_arrived_mainthread)
        self.client = BitalinoClient()
        self.client.configure(address=self.address, sampling_rate=self.fs, analog_channels=self.analog_channels)
        self.btn_save_rr.clicked.connect(self._on_save_rr_clicked)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        self._set_hr("--", "--")
        self._set_rr_count(0)
        
        # --- 結果目錄（\results），若無就建立 ---
        try:
            self.project_root = Path(__file__).resolve().parent.parent
        except Exception:
            self.project_root = Path.cwd()
        self.results_dir = (self.project_root / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # --- 受試者基本資料（從 cfg 讀，沒有就給預設）---
        subj = cfg.get("subject", {})
        self.subject_id   = str(subj.get("id", "RR0000"))   # 例：RR0006
        self.subject_name = str(subj.get("name", ""))       # 例：林測試
        try:
            self.subject_age  = int(subj.get("age", 0))     # 例：62
        except Exception:
            self.subject_age  = 0
        self.subject_sex  = str(subj.get("sex", "U")).upper()  # M / F / U

        # 額外備註欄（可在 cfg 設定）
        self.lead    = str(cfg.get("lead", "Lead II"))
        self.posture = str(cfg.get("posture", ""))
        self.notes   = str(cfg.get("notes", ""))

        # 錄製開始時間（存檔會用）
        self._session_start_iso = None
        self._session_start_t0  = None


        # 電池
        b_cfg = cfg.get("battery", {})
        self.batt_enabled = bool(b_cfg.get("enable", True))
        self.batt_poll_ms = int(b_cfg.get("poll_s", 15)) * 1000
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

    # ── 方向/座標 ───────────────────────────────────────────────
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

    # ── 連線/斷線 ───────────────────────────────────────────────
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

    # ── callbacks ──────────────────────────────────────────────
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

    # ── 開始/停止/暫停/續傳 ─────────────────────────────────────
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

            # 復位濾波/引擎
            self.warmup_left = int(0.4 * self.fs)
            if getattr(self, "ecg_filter", None) is not None and hasattr(self.ecg_filter, "reset_state"):
                try:
                    self.ecg_filter.reset_state()
                except Exception:
                    pass
            self.rr_engine.reset()

            # 清 RR/HR 與畫面
            self._rr_accum.clear()
            self.hr_stable = None
            self._set_hr("--", "--")
            self._set_rr_count(0)

            self._ybuf[:] = 0.0
            self._g_draw = 0
            self._rr_anchor = 0
            self._rr_started = False
            self.curve.setData(self._tbase, self._ybuf)
            self._pos = 0
            self._wrapped_once = False
            self._rwin.clear()
            self.scatter.setData([], [])
            self._apply_direction()
            
            # 紀錄錄製開始時間（檔頭 START_TIME / DURATION 用）
            self._session_start_t0  = time.time()
            self._session_start_iso = datetime.now().astimezone().isoformat(timespec='seconds')

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
            self.status_bar.showMessage("已暫停（保留狀態，可續傳）", 2500)
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
            chunk = max(10, min(self.buf_len // 2, int(self.chunk)))
            self.client.start_acquisition(chunk_size=chunk)
            self._rx_counter = 0; self._rx_last = 0; self._rx_zero_ticks = 0
            self._rx_timer.start()
            self._start_battery_monitor()
            self._is_streaming = True
            self._is_paused = False
            self.status_bar.showMessage("續傳中（狀態已保留）", 2000)
            return True
        except Exception as e:
            self.status_bar.showMessage(f"續傳失敗：{e}", 5000)
            return False

    # ── 主執行緒：接資料 → 濾波/偵測/繪圖 ─────────────────────────
    def _on_arrived_mainthread(self, arr_obj: object):
        self._apply_direction()
        data = np.atleast_2d(np.asarray(arr_obj))
        if data.shape[0] == 0:
            return

        self._rx_counter += data.shape[0]
        if not hasattr(self, "_first_chunk_seen"):
            self._first_chunk_seen = True
            self.status_bar.showMessage(f"✓ 已收到資料：shape={data.shape}", 2500)

        # 取 ECG 欄位
        ecg = data[:, -1].astype(float) if self.ecg_col == -1 else data[:, int(self.ecg_col)].astype(float)

        # 濾波 + 增益
        if self.filter_enabled and self.ecg_filter is not None:
            try:
                y = self.ecg_filter.process(ecg) * self.gain
            except Exception:
                y = ecg * self.gain
        else:
            y = ecg * self.gain

        # 畫圖（掃描/滑動）
        if self.mode == "sweep":
            self._plot_sweep(y)
        else:
            self._plot_sliding(y)

        # 暖機
        n = len(y)
        if self.warmup_left > 0:
            self.warmup_left -= n
            return
        
        if not self._rr_started:
            # 這一包第一筆樣本在畫面上的全域序 = 目前已畫總數 - 本包長度
            self._rr_anchor = self._g_draw - len(y)
            self._rr_started = True

        # ★ 用專用 RR 引擎
        r_disp, rr_new = self.rr_engine.process(y)

        # 紅點：僅視覺（近一窗長度內）
        if r_disp:
            for gi in r_disp:
                self._rwin.append(gi)
            self._prune_rwin_and_update_scatter()

        # RR/HR：完全以 RR 引擎的次取樣結果為準
        if rr_new:
            self._rr_accum.extend(rr_new)
            rr_tail = np.asarray(self._rr_accum[-5:], dtype=float)
            hr_inst = 60_000.0 / float(rr_tail.mean())
            if self.hr_stable is None:
                self.hr_stable = hr_inst
            else:
                self.hr_stable = (1 - self.alpha) * self.hr_stable + self.alpha * hr_inst
            self._set_hr(f"{hr_inst:.0f}", f"{self.hr_stable:.0f}")
            self._set_rr_count(len(self._rr_accum))

    # ── 掃描式 ──────────────────────────────────────────────────
    def _plot_sweep(self, y: np.ndarray):
        L = self._ybuf.size; n = int(y.size)
        if n <= 0 or L == 0:
            return
        pos = self._pos
        if n >= L:
            self._ybuf[:] = y[-L:]; pos = 0; self._wrapped_once = True
        else:
            end = pos + n
            if end <= L:
                self._ybuf[pos:end] = y; pos = end
                if pos == L:
                    pos = 0; self._wrapped_once = True
            else:
                k = L - pos
                self._ybuf[pos:] = y[:k]
                self._ybuf[:n - k] = y[k:]
                pos = (n - k); self._wrapped_once = True
        self._pos = pos
        y_vis = self._ybuf.copy()
        if not self._wrapped_once:
            if self._pos < L: y_vis[self._pos:] = np.nan
        else:
            y_vis[self._pos % L] = np.nan
        self.curve.setData(self._tbase, y_vis, connect='finite')
        self._g_draw += n   # ← 累積畫了幾個樣本

    # ── 滑動窗 ──────────────────────────────────────────────────
    def _plot_sliding(self, y: np.ndarray):
        L = self._ybuf.size; n = int(y.size)
        if n <= 0: return
        if n >= L: self._ybuf[:] = y[-L:]
        else:
            self._ybuf = np.roll(self._ybuf, -n)
            self._ybuf[-n:] = y
        self.curve.setData(self._tbase, self._ybuf)
        self._g_draw += n
        self._prune_rwin_and_update_scatter()
         
    # ── RR 存檔 / HRV（時域） ───────────────────────────────────
    def _on_save_rr_clicked(self):
        if len(self._rr_accum) == 0:
            QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return

        # 檔名：SUBJECT_ID_SEX_AGE_YYYYMMDD.txt
        date_str = datetime.now().strftime("%Y%m%d")
        sex_tag  = (self.subject_sex or "U").upper()
        age_tag  = str(self.subject_age) if self.subject_age else "0"
        fname    = f"{self.subject_id}_{sex_tag}_{age_tag}_{date_str}.txt"
        default_path = (self.results_dir / fname).resolve()

        # 儲存對話框（預設指到 \results）
        fn, _ = QFileDialog.getSaveFileName(
            None, "儲存 RR", str(default_path), "Text Files (*.txt)"
        )
        if not fn:
            return

        # 檔頭 + 整數毫秒 RR
        header = self._compose_rr_header(self._rr_accum)
        rr_lines = "\n".join(str(int(round(v))) for v in self._rr_accum)

        out_path = Path(fn)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        out_path.write_text(header + rr_lines + "\n", encoding="utf-8")
        QMessageBox.information(None, "儲存 RR", f"已儲存：{out_path}")


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
        
    def _compose_rr_header(self, rr_ms: List[float]) -> str:
        rr = np.asarray(rr_ms, dtype=float)
        n  = rr.size

        # 基本統計
        mean_rr = float(rr.mean()) if n else 0.0
        sdnn    = float(rr.std(ddof=1)) if n > 1 else 0.0
        diffs   = np.diff(rr)
        rmssd   = float(np.sqrt(np.mean(diffs * diffs))) if diffs.size else 0.0
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0

        # 錄製時長
        dur_s = 0.0
        if self._session_start_t0 is not None:
            dur_s = max(0.0, time.time() - self._session_start_t0)

        # 濾波資訊
        if self.ecg_filter is None:
            filt_str = "none"
        elif _HAS_ECGFILTER and isinstance(self.ecg_filter, ECGFilterRT):
            try:
                band = tuple(getattr(self.ecg_filter, "band", (0.5, 25.0)))
            except Exception:
                band = (0.5, 25.0)
            notch_on = "ON" if self.notch_enabled else "OFF"
            filt_str = f"bandpass {band[0]:.1f}–{band[1]:.1f} Hz; notch 60 Hz={notch_on}"
        else:
            notch_on = "ON" if self.notch_enabled else "OFF"
            filt_str = f"bandpass 0.67–30.0 Hz; notch 60 Hz={notch_on}"

        # 偵測器參數（從 rr_engine 反推）
        det = self.rr_engine
        try:
            # alpha = exp(-1/(fs*tau)) → tau(ms) = -1000/(fs*ln(alpha))
            env_ms = int(round(-1000.0 / (self.fs * math.log(det.alpha)))) if det.alpha > 0 and det.alpha < 1 else 0
        except Exception:
            env_ms = 0
        search_ms    = int(round(det.search * 1000.0 / self.fs))
        refractoryMs = int(round(det.refractory * 1000.0 / self.fs))
        det_str = f"EMA_env={env_ms}ms; search=±{search_ms}ms; refractory={refractoryMs}ms; min_env=0.0; min_peak=0.0"

        header = [
            "# FORMAT: RR intervals (ms), one per line",
            f"# SUBJECT_ID: {self.subject_id}",
            f"# Name: {self.subject_name}",
            f"# AGE: {self.subject_age}",
            f"# SEX: {self.subject_sex}",
            f"# DEVICE: BITalino (r)evolution; fs={self.fs} Hz; channel=A1",
            f"# LEAD: {self.lead}",
            f"# START_TIME: {self._session_start_iso or ''}",
            f"# DURATION_S: {dur_s:.1f}",
            f"# RR_COUNT: {n}",
            "# UNITS: ms",
            "# DATA_STATUS: raw-detected (no ectopic removal)",
            f"# FILTER: {filt_str}",
            f"# DETECTOR: {det_str}",
            "# CLEANING: outliers <300 or >2000 ms removed=NO",
            f"# METRICS: meanRR={mean_rr:.1f} ms; meanHR={mean_hr:.1f} bpm; SDNN={sdnn:.1f} ms; RMSSD={rmssd:.1f} ms",
            f"# POSTURE: {self.posture}",
            f"# NOTES: {self.notes}",
            ""
        ]
        return "\n".join(header)


    # ── UI 小工具 ───────────────────────────────────────────────
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

    def _prune_rwin_and_update_scatter(self):
        if not self._rwin:
            self.scatter.setData([], [])
            return

        L = self.buf_len
        if L <= 0:
            self.scatter.setData([], [])
            return

        # 僅保留「一個視窗長度內」的 R（避免點落在已被覆寫的資料上）
        # 改用 RR 引擎的全域索引（不再使用 self.det）
        g_now = self.rr_engine.global_index
        g_min = max(0, g_now - L)
        while self._rwin and self._rwin[0] < g_min:
            self._rwin.popleft()
        if not self._rwin:
            self.scatter.setData([], [])
            return

        # 1) 以錨點將 RR 引擎的全域索引 gi 映射到畫面 ring-buffer 索引
        # 2) 再在 ±6 樣本內尋找「局部最大值」作為頂點微調（強化視覺貼合）
        xs, ys = [], []
        lag = int(self._rpeak_vis_lag_samples)
        search_half = 6  # 可視需要調整 4~8

        for gi in self._rwin:
            # 基準索引（可能有 1~2 點誤差）
            base = (self._rr_anchor + gi + lag) % L

            # 在 ±search_half 範圍找局部最大值（正 R）
            idxs = [(base + d) % L for d in range(-search_half, search_half + 1)]
            local = self._ybuf[idxs]
            off = int(np.argmax(local))   # 正 R：找最大；若負 R 改用 argmin
            best = idxs[off]

            xs.append(float(self._tbase[best]))
            ys.append(float(self._ybuf[best]))

        self.scatter.setData(xs, ys)

    # ── 吞吐監看 ───────────────────────────────────────────────
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

    # ── 電池 ───────────────────────────────────────────────────
    def _battery_percent_from_raw(self, raw: float) -> float:
        r0, r1 = self.batt_raw_min, self.batt_raw_max
        if r1 <= r0: return 0.0
        pct = 1.0 + (float(raw) - r0) * (98.0 / (r1 - r0))
        return max(0.0, min(100.0, pct))

    def _query_battery_percent(self):
        dev = getattr(self.client, "device", None)
        if dev is None: return None
        try:
            st = dev.state()
            raw = st.get("battery") if isinstance(st, dict) else None
            if raw is None: return None
            return self._battery_percent_from_raw(raw)
        except Exception:
            return None

    def _set_batt_label(self, pct: Optional[float]):
        if not self.batt_enabled: return
        if pct is None:
            self.batt_label.setText("🔋 --%"); self.batt_label.setStyleSheet(""); return
        self.batt_label.setText(f"🔋 {pct:0.0f}%")
        if pct <= self.batt_crit_pct:
            self.batt_label.setStyleSheet("color:#e53935;")
        elif pct <= self.batt_low_pct:
            self.batt_label.setStyleSheet("color:#fb8c00;")
        else:
            self.batt_label.setStyleSheet("")

    def _on_batt_tick(self):
        if getattr(self.client, "is_acquiring", False): return
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