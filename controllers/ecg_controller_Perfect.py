# controllers/ecg_controller.py
"""
ECGController —— BITalino 串流 → 濾波 → R 峰 → HR/HRV → 繪圖
教學就緒版（掃描式左→右、濾波/Notch 切換、電池顯示〈idle〉、資料吞吐監看）

相依：
- PyQt6, pyqtgraph, numpy
- bitalino_helpers.BitalinoClient  (專案內的連線/串流包裝)
- processing/filters.py（若有，使用 ECGFilterRT；沒有則內建簡化濾波）
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QLabel, QMessageBox, QFileDialog

from bitalino_helpers import BitalinoClient

# ─────────────────────────────────────────────────────────────
# 濾波：優先使用 processing/filters.ECGFilterRT；缺少時使用 fallback
# ─────────────────────────────────────────────────────────────
_HAS_ECGFILTER = False
try:
    from processing.filters import ECGFilterRT, EnhancedBandpass  # 專業帶通＋可選 Notch
    _HAS_ECGFILTER = True
except Exception:
    # 內建簡化濾波（仍可即時演示）：高通(移除基線) → 低通 →（若有 SciPy）Notch + 中值濾波
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
                s = a * s + (1 - a) * v
                y[i] = s
            self.s = s
            return y

    class EnhancedBandpass:
        """簡化帶通（0.67–30Hz）+（可選）50/60Hz Notch；若有 SciPy 再做 medfilt。"""

        def __init__(self, fs: int, hp=0.67, lp=30.0, notch_freq: float = 0.0):
            self.fs = int(fs)
            self.hp_lp = _OnePoleLPF(fs, hp)   # 當作 baseline 做高通
            self.lp_lp = _OnePoleLPF(fs, lp)
            self.use_notch = _HAS_SCIPY and notch_freq and notch_freq > 0
            if self.use_notch:
                b, a = _sig.iirnotch(w0=notch_freq/(fs/2.0), Q=30.0)
                self.sos_notch = _sig.tf2sos(b, a)
                self.zi_notch = _sig.sosfilt_zi(self.sos_notch)
            else:
                self.sos_notch = None
                self.zi_notch = None

        def process(self, x: np.ndarray) -> np.ndarray:
            baseline = self.hp_lp.filt_vec(x)
            y = x - baseline
            if self.use_notch:
                y, self.zi_notch = _sig.sosfilt(
                    self.sos_notch, y, zi=self.zi_notch)
            y = self.lp_lp.filt_vec(y)
            if _HAS_SCIPY and len(y) >= 5:
                y = _sig.medfilt(y, kernel_size=5)
            return y

        def set_notch_enabled(self, enabled: bool):  # 與專業版介面對齊
            self.use_notch = bool(enabled)

        def reset_state(self):
            self.hp_lp.s = 0.0
            self.lp_lp.s = 0.0

# ─────────────────────────────────────────────────────────────
# R 峰偵測（輕量、即時友善） & HRV 時域
# ─────────────────────────────────────────────────────────────


class ECGDetector:
    """微分→平方→移動整流→動態門檻 + 不應期；適合串流逐批處理。"""

    def __init__(self, fs: int):
        self.fs = int(fs)
        self._integ_buf = np.zeros(max(1, int(0.15 * fs)), dtype=float)
        self._integ_idx = 0
        self._thr = 0.0
        self._last_peak_i = -10_000
        self._global_i = 0
        self.r_indices: List[int] = []
        self.rr_ms: List[float] = []

    def process(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        # 微分 + 平方
        if x.size == 1:
            diff = np.array([x[0]], dtype=float)
        else:
            diff = np.empty_like(x, dtype=float)
            diff[0] = x[1] - x[0]
            diff[1:] = x[1:] - x[:-1]
        sq = diff * diff
        # 150ms 移動整流（簡潔環形平均）
        w = self._integ_buf.size
        integ = np.empty_like(sq, dtype=float)
        for i, v in enumerate(sq):
            self._integ_buf[self._integ_idx] = v
            self._integ_idx = (self._integ_idx + 1) % w
            integ[i] = float(self._integ_buf.mean())
        # 動態門檻 + 不應期
        self.r_indices.clear()
        self.rr_ms.clear()
        med = float(np.median(integ))
        std = float(np.std(integ))
        target_thr = med + 0.8 * std
        self._thr = 0.9 * self._thr + 0.1 * target_thr
        refr = int(0.25 * self.fs)  # 250 ms
        for i in range(1, len(integ) - 1):
            gi = self._global_i + i
            if gi - self._last_peak_i < refr:
                continue
            if integ[i-1] < integ[i] >= integ[i+1] and integ[i] > self._thr:
                self.r_indices.append(gi)
                if self._last_peak_i >= 0:
                    rr = (gi - self._last_peak_i) * 1000.0 / self.fs
                    if 300.0 <= rr <= 2000.0:
                        self.rr_ms.append(rr)
                self._last_peak_i = gi
        self._global_i += len(integ)


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
# Qt 資料橋：把背景 thread 的資料安全丟回主執行緒
# ─────────────────────────────────────────────────────────────


class DataBridge(QObject):
    arrived = pyqtSignal(object)  # numpy.ndarray

# ─────────────────────────────────────────────────────────────
# 主控制器（掃描式 LTR 固定 0~N 秒視窗）
# ─────────────────────────────────────────────────────────────


class ECGController:
    """
    把 UI 與 BitalinoClient 串起來：
      - connect / disconnect / start / stop
      - 背景串流回呼 → Qt 訊號 → 主執行緒：濾波→R 峰→HR→繪圖（掃描式左→右）
      - 濾波/Notch 即時切換、RR 存檔 / HRV 時域、電池顯示（idle 才讀）、資料吞吐監看
    """

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        lbl_rt_hr,
        lbl_stable_hr,
        status_bar,
        btn_save_rr,
        btn_analyze,
        cfg: Dict[str, Any],
    ):
        # --- UI 控件 ---
        self.plot = plot_widget
        self.lbl_rt = lbl_rt_hr
        self.lbl_stable = lbl_stable_hr
        self.status_bar = status_bar
        self.btn_save_rr = btn_save_rr
        self.btn_analyze = btn_analyze

        # --- 基本設定 ---
        self.fs = int(cfg.get("sampling_rate", 1000))

        # --- BITalino 裝置設定 ---
        ch = cfg.get("analog_channels", [1])
        if not isinstance(ch, list):
            ch = [int(ch)]
        self.analog_channels: List[int] = ch
        self.address: Optional[str] = cfg.get("address") or None

        # --- 濾波器設定（有 ECGFilterRT 用它，否則用簡化版）---
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
            # fallback：Notch 由 EnhancedBandpass(notch_freq) 控制
            notch_freq = float(f_cfg.get("notch", 0.0))
            self.ecg_filter = EnhancedBandpass(
                self.fs, hp=0.67, lp=30.0, notch_freq=notch_freq)
            self._use_fallback = True
        else:
            self.ecg_filter = None
            self._use_fallback = False

        # Filter / Notch 旗標（供 UI 勾選切換）
        self.filter_enabled = bool(f_cfg.get("enable", True))
        self.notch_enabled = bool(
            f_cfg.get("notch", 60.0) and not self._use_fallback)
        if self.ecg_filter is not None and hasattr(self.ecg_filter, "set_notch_enabled"):
            try:
                self.ecg_filter.set_notch_enabled(self.notch_enabled)
            except Exception:
                pass

        # --- 繪圖/緩衝（掃描式 LTR） ---
        plot_cfg = cfg.get("plot", {})
        self.seconds_window = int(plot_cfg.get("seconds", 10))
        self.buf_len = max(10, int(self.seconds_window * self.fs))
        self.gain = float(plot_cfg.get("gain", 1.5))
        self.ecg_col = int(plot_cfg.get("ecg_col", -1))  # -1=最後一欄
        self.chunk = int(plot_cfg.get("chunk", 100))

        # 固定視窗資料緩衝（掃描式 LTR）
        self._ybuf = np.full(self.buf_len, np.nan,
                             dtype=float)       # 空白 = NaN
        self._tbase = np.linspace(0.0, float(
            self.seconds_window), self.buf_len, endpoint=False)
        self._write_pos = 0                                           # 畫筆位置（index）

        # 單一曲線
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        self.curve.setDownsampling(auto=True)
        self.curve.setClipToView(True)
        self.curve.setData(self._tbase, self._ybuf)                   # 先畫空畫面

        # 圖面樣式與座標軸（永遠左→右）
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.showGrid(x=True, y=True)

        vb = self.plot.getViewBox()
        vb.setDefaultPadding(0.25)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        vb.setXRange(0, self.seconds_window, padding=0)
        # ★ 關鍵：不反轉
        vb.invertX(False)

        # --- 偵測/顯示 ---
        self.det = ECGDetector(self.fs)
        self.warmup_left = int(0.8 * self.fs)   # 暖機（先不做 R 偵測）
        self.alpha = 0.12                       # 穩定 HR 的 EMA 係數
        self.hr_stable: Optional[float] = None
        self._rr_accum: List[float] = []

        # --- 橋接：把背景執行緒資料送回主執行緒 ---
        self.bridge = DataBridge()
        self.bridge.arrived.connect(self._on_arrived_mainthread)

        # --- Bitalino Client ---
        self.client = BitalinoClient()
        self.client.configure(
            address=self.address,
            sampling_rate=self.fs,
            analog_channels=self.analog_channels,
        )

        # --- UI 綁定 ---
        self.btn_save_rr.clicked.connect(self._on_save_rr_clicked)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)

        # 初始 HR 顯示
        self._set_hr("--", "--")

        # --- 電池監測（只在 idle 時輪詢） ---
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

        # --- 資料吞吐監看（每秒顯示 samples/s；連 2 秒 0/s 就提醒） ---
        self.rx_label = QLabel("📡 0/s")
        self.status_bar.addPermanentWidget(self.rx_label)
        self._rx_counter = 0
        self._rx_last = 0
        self._rx_zero_ticks = 0
        self._rx_timer = QTimer(self.bridge)
        self._rx_timer.setInterval(1000)
        self._rx_timer.timeout.connect(self._on_rx_tick)

        # 流程旗標
        self._is_streaming = False

    # ── 濾波/Notch 切換（給 UI 勾選） ─────────────────────────────
    def set_filter_enabled(self, enabled: bool):
        self.filter_enabled = bool(enabled)
        self.status_bar.showMessage(
            f"濾波：{'開' if self.filter_enabled else '關'}", 1500)

    def set_notch_enabled(self, enabled: bool):
        self.notch_enabled = bool(enabled)
        if self.ecg_filter is not None and hasattr(self.ecg_filter, "set_notch_enabled"):
            try:
                self.ecg_filter.set_notch_enabled(self.notch_enabled)
                self.status_bar.showMessage(
                    f"Notch 60Hz：{'開' if self.notch_enabled else '關'}", 1500)
            except Exception:
                self.status_bar.showMessage("Notch 切換失敗（濾波器不支援）", 3000)
        else:
            self.status_bar.showMessage(
                "目前濾波器不支援 Notch（未安裝 scipy 或 notch=0）", 3000)

    # ── 連線 / 斷線 ───────────────────────────────────────────────
    def connect_device(self, address: Optional[str] = None, retries: int = 3) -> bool:
        """建立連線；idle 時先讀一次電池，再啟動電池輪詢（串流時不讀 state）。"""
        try:
            self.status_bar.showMessage("連線中…")
            if address:
                self.address = address
                try:
                    self.client.configure(address=address)
                except Exception:
                    pass
            self.client.connect(retries=retries)

            # 連線成功且仍 idle → 先讀一次電池
            try:
                self._set_batt_label(self._query_battery_percent())
                if self.batt_enabled and hasattr(self.client, "device") and self.client.device:
                    # 設定裝置低電量門檻（非必要）
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

    # ── 開始 / 停止 串流 ─────────────────────────────────────────
    def start_stream(self) -> bool:
        """
        啟動串流（掃描式 LTR）：
        - 防重入
        - 綁定資料/錯誤回呼
        - 確保已連線
        - 重置掃描式緩衝並顯示空畫面（NaN 斷筆）
        - 啟動擷取、吞吐監看與（idle 才有效的）電池輪詢
        """
        if self._is_streaming:
            self.status_bar.showMessage("已在擷取中", 1500)
            return True

        try:
            # 背景 → 主執行緒
            def _on_data(arr):
                try:
                    self.bridge.arrived.emit(np.asarray(arr))
                except Exception as ex:
                    print("UI dispatch error:", ex)
            self.client.data_callback = _on_data

            # 讀取執行緒錯誤回呼（若支援）
            if hasattr(self.client, "on_error"):
                def _on_err(e):
                    self._is_streaming = False
                    try:
                        self.client.stop_acquisition()
                    except Exception:
                        pass
                    self._stop_battery_monitor()
                    self._rx_timer.stop()
                    self.status_bar.showMessage(f"資料擷取中斷：{e}", 8000)
                self.client.on_error = _on_err

            # 確保已連線
            if not getattr(self.client, "is_connected", False):
                self.client.connect(retries=3)

            # 復位濾波/偵測
            self.warmup_left = int(0.8 * self.fs)
            if getattr(self, "ecg_filter", None) is not None and hasattr(self.ecg_filter, "reset_state"):
                try:
                    self.ecg_filter.reset_state()
                except Exception:
                    pass
            self.det = ECGDetector(self.fs)

            # 重置掃描式緩衝並清畫面
            self._ybuf[:] = np.nan
            self._write_pos = 0
            self.curve.setData(self._tbase, self._ybuf)

            # 座標軸鎖定 LTR
            vb = self.plot.getViewBox()
            vb.invertX(False)
            vb.setXRange(0, self.seconds_window, padding=0)

            # 啟動擷取
            chunk = max(10, min(self.buf_len // 2, int(self.chunk)))
            self.client.start_acquisition(chunk_size=chunk)
            self._is_streaming = True

            # 吞吐監看
            self._rx_counter = 0
            self._rx_last = 0
            self._rx_zero_ticks = 0
            self._rx_timer.start()

            self.status_bar.showMessage(
                f"擷取中：fs={self.fs}, ch={self.analog_channels}, chunk={chunk}", 3500
            )

            # 電池監看（注意：擷取中 _on_batt_tick 會自動略過 state()）
            self._start_battery_monitor()
            return True

        except Exception as e:
            self._is_streaming = False
            try:
                self.client.stop_acquisition()
            except Exception:
                pass
            self._rx_timer.stop()
            try:
                QMessageBox.critical(None, "開始擷取失敗", str(e))
            except Exception:
                pass
            self.status_bar.showMessage(f"開始擷取失敗：{e}", 6000)
            return False

    def stop_stream(self):
        try:
            self._is_streaming = False
            self._rx_timer.stop()
            self.client.stop_acquisition()
            # 停下來後（idle）主動刷新一次電池
            self._on_batt_tick()
            self.status_bar.showMessage("已停止擷取", 1800)
        except Exception as e:
            self.status_bar.showMessage(f"停止擷取失敗：{e}", 4000)

    # ── 主執行緒：接資料 → 濾波/偵測/繪圖 ─────────────────────────
    def _on_arrived_mainthread(self, arr_obj: object):
        data = np.asarray(arr_obj)
        if data.ndim != 2 or data.shape[0] == 0:
            return

        # 吞吐統計（📡）
        self._rx_counter += data.shape[0]

        # 取 ECG 欄位
        ecg = data[:, -1].astype(float) if self.ecg_col == - \
            1 else data[:, int(self.ecg_col)].astype(float)

        # 濾波 + 顯示增益
        if self.filter_enabled and self.ecg_filter is not None:
            try:
                y = self.ecg_filter.process(ecg) * self.gain
            except Exception:
                y = ecg * self.gain
        else:
            y = ecg * self.gain

        # 掃描式 LTR 繪圖
        self._plot_sweep_ltr(y)

        # 暖機期間不做 R 偵測
        n = len(y)
        if self.warmup_left > 0:
            self.warmup_left -= n
            return

        # R 峰 → RR → HR
        self.det.process(y)
        if self.det.rr_ms:
            self._rr_accum.extend(self.det.rr_ms)
            rr_arr = np.asarray(self.det.rr_ms[-5:], dtype=float)
            hr_inst = 60_000.0 / float(rr_arr.mean())
            if self.hr_stable is None:
                self.hr_stable = hr_inst
            else:
                self.hr_stable = (1 - self.alpha) * \
                    self.hr_stable + self.alpha * hr_inst
            self._set_hr(f"{hr_inst:.0f}", f"{self.hr_stable:.0f}")

    # ── 掃描式（左→右）畫圖：固定 0~seconds，不平移 ─────────────────
    def _plot_sweep_ltr(self, y: np.ndarray):
        """
        專業心電圖掃描式（左→右）：
        - 視窗固定 0~seconds_window 不動
        - 畫筆自左向右寫入緩衝；到尾端就回 0 繼續覆寫舊點
        - 在畫筆位置插入 NaN 斷點，避免最後一點連到起點產生水平線
        """
        L = self._ybuf.size
        n = int(y.size)
        if n <= 0:
            return

        pos = int(self._write_pos)

        if n >= L:
            # 新資料太多，直接取最後 L 點覆蓋整窗，畫筆回到 0
            self._ybuf[:] = y[-L:]
            pos = 0
        else:
            end = pos + n
            if end <= L:
                self._ybuf[pos:end] = y
                pos = end
            else:
                k = L - pos
                self._ybuf[pos:] = y[:k]
                self._ybuf[:n - k] = y[k:]
                pos = (n - k)

        self._write_pos = pos % L

        # 為了避免在回圈接頭處出現一條連線，用 NaN 打斷
        disp = self._ybuf.copy()
        disp[self._write_pos] = np.nan

        # 永遠以 0→N 秒的時間軸顯示（不反轉、不平移）
        self.curve.setData(self._tbase, disp)

    # ── RR 存檔 / HRV ───────────────────────────────────────────
    def _on_save_rr_clicked(self):
        if len(self._rr_accum) == 0:
            QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn, _ = QFileDialog.getSaveFileName(
            None, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)")
        if not fn:
            return
        Path(fn).write_text(
            "\n".join(f"{v:.1f}" for v in self._rr_accum), encoding="utf-8")
        QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    def _on_analyze_clicked(self):
        res = compute_time_domain(self._rr_accum)
        if res is None:
            QMessageBox.information(None, "HRV 分析", "RR 數量不足，請先擷取 RR。")
            return
        msg = (
            f"RR 數量：{res.count}\n"
            f"Mean RR：{res.mean_rr:.1f} ms\n"
            f"SDNN：{res.sdnn:.1f} ms\n"
            f"RMSSD：{res.rmssd:.1f} ms\n"
            f"Mean HR：{res.mean_hr:.1f} bpm\n"
        )
        QMessageBox.information(None, "HRV（時域）", msg)

    # ── UI 小工具 ───────────────────────────────────────────────
    def _set_hr(self, rt: str, stable: str):
        try:
            self.lbl_rt.setText(f"即時心跳：{rt} bpm")
            self.lbl_stable.setText(f"穩定心跳：{stable} bpm")
        except Exception:
            pass

    # ── 資料吞吐監看 ───────────────────────────────────────────
    def _on_rx_tick(self):
        now = self._rx_counter
        rate = max(0, now - self._rx_last)
        self._rx_last = now
        self.rx_label.setText(f"📡 {rate}/s")
        if self._is_streaming:
            if rate == 0:
                self._rx_zero_ticks += 1
                if self._rx_zero_ticks >= 2:
                    self.status_bar.showMessage(
                        "⚠ 未收到資料：請確認 COM/電源/取樣已啟動與接線", 4000)
            else:
                self._rx_zero_ticks = 0

    # ── 電池（idle 時才讀 state()） ────────────────────────────
    def _battery_percent_from_raw(self, raw: float) -> float:
        r0, r1 = self.batt_raw_min, self.batt_raw_max
        if r1 <= r0:
            return 0.0
        pct = 1.0 + (float(raw) - r0) * (98.0 / (r1 - r0))  # 約 1~99%
        return max(0.0, min(100.0, pct))

    def _query_battery_percent(self):
        dev = getattr(self.client, "device", None)
        if dev is None:
            return None
        try:
            st = dev.state()  # ★ 只能在 idle 呼叫
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
            self.batt_label.setStyleSheet("color:#e53935;")   # 紅
        elif pct <= self.batt_low_pct:
            self.batt_label.setStyleSheet("color:#fb8c00;")   # 橘
        else:
            self.batt_label.setStyleSheet("")

    def _on_batt_tick(self):
        # 串流時 BITalino 不允許 state()（會丟 "The device is not idle."）
        if getattr(self.client, "is_acquiring", False):
            return
        pct = self._query_battery_percent()
        self._set_batt_label(pct)
        if pct is not None and pct <= self.batt_crit_pct:
            self.status_bar.showMessage("電量過低：請儘快充電", 2500)

    def _start_battery_monitor(self):
        if self.batt_enabled:
            self._batt_timer.start()
            self._on_batt_tick()  # 立即更新一次

    def _stop_battery_monitor(self):
        try:
            self._batt_timer.stop()
            self._set_batt_label(None)
        except Exception:
            pass
