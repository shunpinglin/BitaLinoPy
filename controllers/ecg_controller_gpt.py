# controllers/ecg_controller.py
"""
ECGController
- Device lifecycle: connect/disconnect, start/stop streaming
- Real-time pipeline: ring buffer -> (optional) filtering -> plot update
- Heart rate (instant/stable) compute, save RR, trigger HRV analysis hook
- Threading/timers to ensure UI safety
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal

from bitalino_helpers import BitalinoClient

# --- 嘗試用高品質濾波（scipy + ECGFilterRT）；失敗則 fallback 到內建簡化濾波 ---
try:
    from processing.filters import ECGFilterRT  # 需要 scipy
    _HAS_ECGFILTER = True
except Exception:
    _HAS_ECGFILTER = False


# ========== 內建簡化濾波（沒裝 scipy 時使用） ==========
class OnePoleLPF:
    def __init__(self, fs: int, fc: float):
        a = float(np.exp(-2.0 * np.pi * fc / fs))
        self.a = a
        self.s = 0.0

    def filt_vec(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=float)
        s = self.s
        a = self.a
        xv = x.astype(float, copy=False)
        for i, v in enumerate(xv):
            s = a * s + (1 - a) * v
            y[i] = s
        self.s = s
        return y


class EnhancedBandpass:
    """簡化版：基線移除 + 低通（無零相位、串流友善；教學/無 scipy 時可用）"""

    def __init__(self, fs: int, hp=0.67, lp=30.0):
        self.hp_lp = OnePoleLPF(fs, hp)  # 用低通估 baseline，再相減=高通
        self.lp_lp = OnePoleLPF(fs, lp)

    def process(self, x: np.ndarray) -> np.ndarray:
        base = self.hp_lp.filt_vec(x)
        y = x - base
        y = self.lp_lp.filt_vec(y)
        return y


# ========== 輕量 R 峰偵測（即時串流友善） ==========
class ECGDetector:
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
            diff = np.array([0.0], dtype=float)  # 單點差分視為 0，避免尖峰
        else:
            diff = np.empty_like(x, dtype=float)
            diff[0] = x[1] - x[0]
            diff[1:] = x[1:] - x[:-1]
        sq = diff * diff

        # 150 ms 移動整流（用環形緩衝取平均）
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
            if integ[i - 1] < integ[i] >= integ[i + 1] and integ[i] > self._thr:
                self.r_indices.append(gi)
                if self._last_peak_i >= 0:
                    rr = (gi - self._last_peak_i) * 1000.0 / self.fs
                    if 300.0 <= rr <= 2000.0:
                        self.rr_ms.append(rr)
                self._last_peak_i = gi

        self._global_i += len(integ)


# ========== 時域 HRV ==========
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


# ========== Qt 資料橋 ==========
class DataBridge(QObject):
    """將背景執行緒（BitalinoClient）取得的資料回送到主執行緒"""
    arrived = pyqtSignal(object)  # numpy.ndarray


# ========== 主控制器 ==========
class ECGController:
    """
    把 UI 與 BitalinoClient 接起來：
      - connect_device / disconnect_device
      - start_stream / stop_stream
      - 背景串流回呼 → Qt 訊號 → 主執行緒濾波/偵測/繪圖
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
                band=tuple(f_cfg.get("band", [0.5, 25.0])),
                notch=float(f_cfg.get("notch", 60.0)),
                order=int(f_cfg.get("order", 4)),
                q=float(f_cfg.get("q", 30.0)),
            )
            self._use_fallback = False
        elif f_cfg.get("enable", True):
            self.ecg_filter = EnhancedBandpass(self.fs, hp=0.67, lp=30.0)
            self._use_fallback = True
        else:
            self.ecg_filter = None
            self._use_fallback = False

        # --- 繪圖/緩衝設定 ---
        plot_cfg = cfg.get("plot", {})
        self.seconds_window = int(plot_cfg.get("seconds", 10))
        self.buf_len = max(10, int(self.seconds_window * self.fs))
        self.gain = float(plot_cfg.get("gain", 1.5))
        self.ecg_col = int(plot_cfg.get("ecg_col", -1))   # -1=最後一欄
        self.chunk = int(plot_cfg.get("chunk", 100))

        # 視覺模式
        self.sweep_mode = bool(plot_cfg.get(
            "sweep", True))          # True=掃描式（像醫院）
        self.direction = str(plot_cfg.get(
            "direction", "ltr")).lower()  # 只在滑動窗時生效

        # ring buffer 與狀態
        self._ybuf = np.zeros(self.buf_len, dtype=float)
        self._xpos = 0                    # 寫入位置（樣本 index，環形）
        self._sample_counter = 0          # 全域樣本計數（不回捲）
        self._prev_pos = 0                # 掃描模式用游標

        # 固定時間軸（秒）：0 → seconds_window
        self._tbase = np.linspace(0.0, float(
            self.seconds_window), self.buf_len, endpoint=False)

        # 建立曲線（單一曲線即可；掃描/滑動共用）
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        self.curve.setDownsampling(auto=True)
        self.curve.setClipToView(True)

        # 圖面樣式
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.showGrid(x=True, y=True)
        self.plot.getViewBox().setXRange(0, self.seconds_window, padding=0)

        # ViewBox：Y 自動、X 由我們控制；掃描模式一律不反轉
        vb = self.plot.getViewBox()
        vb.setDefaultPadding(0.15)
        vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        vb.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
        if self.sweep_mode:
            vb.invertX(False)                       # 掃描模式：畫筆往右
        else:
            vb.invertX(self.direction == "rtl")     # 滑動窗：依 direction

        # --- 偵測/顯示 ---
        self.det = ECGDetector(self.fs)
        self.warmup_left = int(0.8 * self.fs)
        self.alpha = 0.12
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

    # 允許帶入 address；若有就先 re-config 再連線

    def connect_device(self, address: Optional[str] = None, tries: int = 3, wait_s: float = 1.0) -> bool:
        """
        嘗試連線 BITalino；可帶入新的 address（'COM16' 或 'AA:BB:..'）
        回傳 True/False
        """
        if address:
            self.address = address
            self.client.configure(
                address=self.address,
                sampling_rate=self.fs,
                analog_channels=self.analog_channels,
            )
        last_err = None
        for i in range(1, tries + 1):
            try:
                self.status_bar.showMessage(f"連線中…（第 {i}/{tries} 次）")
                if not self.client.is_connected:
                    self.client.connect()
                self.status_bar.showMessage("BITalino 連線成功", 3000)
                return True
            except Exception as e:
                last_err = e
                time.sleep(wait_s)
        self.status_bar.showMessage(f"連線失敗（已重試 {tries} 次）：{last_err}", 6000)
        return False

    def disconnect_device(self):
        try:
            self.stop_stream()
            self.client.close()
            self.status_bar.showMessage("已斷線", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"斷線錯誤：{e}")

    def start_stream(self):
        try:
            # 重置繪圖視窗狀態（不用 NaN，避免某些版本不畫）
            self._ybuf.fill(0.0)
            self._xpos = 0
            self._prev_pos = 0
            self._sample_counter = 0
            self.hr_stable = None
            self._rr_accum.clear()
            self.warmup_left = int(0.8 * self.fs)

            # 背景回呼 → Qt 訊號
            def _on_data(arr: np.ndarray):
                self.bridge.arrived.emit(arr)

            self.client.data_callback = _on_data

            if not self.client.is_connected:
                self.client.connect()
            self.client.start_acquisition(chunk_size=self.chunk)
            self.status_bar.showMessage(
                f"擷取中：fs={self.fs}, ch={self.analog_channels}, chunk={self.chunk}", 3000
            )
        except Exception as e:
            self.status_bar.showMessage(f"開始擷取失敗：{e}")

    def stop_stream(self):
        try:
            self.client.stop_acquisition()
            self.status_bar.showMessage("已停止擷取", 2000)
        except Exception as e:
            self.status_bar.showMessage(f"停止擷取失敗：{e}")

    # ------------ 資料處理（主執行緒）------------
    def _on_arrived_mainthread(self, arr_obj: object):
        data = np.asarray(arr_obj)
        if data.ndim != 2 or data.shape[0] == 0:
            return

        # 取 ECG 欄位（預設最後一欄），確保索引合法
        col = int(self.ecg_col)
        if col < 0:
            col = data.shape[1] + col  # -1 -> 最末欄
        col = max(0, min(col, data.shape[1] - 1))
        ecg = data[:, col].astype(float, copy=False)

        # 濾波 + 增益
        y = self.ecg_filter.process(
            ecg) if self.ecg_filter is not None else ecg
        y = y * float(self.gain)

        # === 寫入環形緩衝 ===
        n = int(y.size)
        if n <= 0:
            return
        p = self._xpos % self.buf_len
        if n >= self.buf_len:
            self._ybuf[:] = y[-self.buf_len:]
            self._xpos += self.buf_len
            self._sample_counter += self.buf_len
        else:
            end = p + n
            if end <= self.buf_len:
                self._ybuf[p:end] = y
            else:
                k = self.buf_len - p
                self._ybuf[p:] = y[:k]
                self._ybuf[: (n - k)] = y[k:]
            self._xpos += n
            self._sample_counter += n

        # === 繪圖 ===
        if self.sweep_mode:
            # 掃描模式：畫筆從左→右；右側保持空白；到右緣後清空再從左開始
            pos = self._xpos % self.buf_len

            # wrap：游標從右邊回到左邊 → 清空畫面
            if pos < self._prev_pos:
                self.curve.setData([], [])   # 清空一次，比畫 NaN 更保險
            self._prev_pos = pos

            # 只畫 0..pos 的有效段（右邊自然空白）
            L = int(min(pos + 1, self.buf_len))
            if L > 0:
                self.curve.setData(self._tbase[:L], self._ybuf[:L])
            else:
                self.curve.setData([], [])

            # 視窗固定在 0..seconds_window
            self.plot.getViewBox().setXRange(0, self.seconds_window, padding=0.0)

        else:
            # 滑動窗（右貼齊最新、整窗往左捲）
            roll = self._xpos % self.buf_len
            y_vis = np.roll(self._ybuf, -roll)
            right_t = self._sample_counter / float(self.fs)
            left_t = max(0.0, right_t - float(self.seconds_window))
            x_vis = np.linspace(left_t, right_t, self.buf_len, endpoint=False)
            self.curve.setData(x_vis, y_vis)
            self.plot.getViewBox().setXRange(left_t, right_t, padding=0.0)

        # === 暖機不做 R 偵測 ===
        if self.warmup_left > 0:
            self.warmup_left -= n
            return

        # === R 峰 → RR → HR ===
        self.det.process(y)
        if self.det.rr_ms:
            self._rr_accum.extend(self.det.rr_ms)
            rr_arr = np.asarray(self.det.rr_ms[-5:], dtype=float)
            if rr_arr.size > 0:
                mean_rr = float(rr_arr.mean())
                if mean_rr > 0:
                    hr_inst = 60_000.0 / mean_rr
                    if np.isfinite(hr_inst):
                        if self.hr_stable is None:
                            self.hr_stable = hr_inst
                        else:
                            self.hr_stable = (1 - self.alpha) * \
                                self.hr_stable + self.alpha * hr_inst
                        self._set_hr(f"{hr_inst:.0f}", f"{self.hr_stable:.0f}")

    # ------------ RR 存檔 & HRV ------------

    def _on_save_rr_clicked(self):
        from PyQt6 import QtWidgets
        if len(self._rr_accum) == 0:
            QtWidgets.QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)"
        )
        if not fn:
            return
        Path(fn).write_text(
            "\n".join(f"{v:.1f}" for v in self._rr_accum), encoding="utf-8")
        QtWidgets.QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    def _on_analyze_clicked(self):
        from PyQt6 import QtWidgets
        res = compute_time_domain(self._rr_accum)
        if res is None:
            QtWidgets.QMessageBox.information(
                None, "HRV 分析", "RR 數量不足，請先擷取 RR。")
            return
        msg = (
            f"RR 數量：{res.count}\n"
            f"Mean RR：{res.mean_rr:.1f} ms\n"
            f"SDNN：{res.sdnn:.1f} ms\n"
            f"RMSSD：{res.rmssd:.1f} ms\n"
            f"Mean HR：{res.mean_hr:.1f} bpm\n"
        )
        QtWidgets.QMessageBox.information(None, "HRV（時域）", msg)

    # ------------ 小工具 ------------
    def _set_hr(self, rt: str, stable: str):
        try:
            self.lbl_rt.setText(f"即時心跳：{rt} bpm")
            self.lbl_stable.setText(f"穩定心跳：{stable} bpm")
        except Exception:
            pass
