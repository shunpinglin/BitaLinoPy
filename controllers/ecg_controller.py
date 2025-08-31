# controllers/ecg_controller.py
from __future__ import annotations
"""
ECGController —— BITalino 串流 → 濾波 → R 峰 → HR/HRV → 繪圖（含自動回主執行緒）
相依：PyQt6, pyqtgraph, numpy；裝置介面由 bitalino_helpers.BitalinoClient 提供
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from bitalino_helpers import BitalinoClient

# ---- 可選：SciPy（有的話就加強濾波；沒有也能跑）----
try:
    import scipy.signal
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    print("⚠ 未安裝 SciPy，將使用簡化濾波。建議：pip install scipy")


# ========== 濾波器（簡化 + 加強兼容） ==========
class OnePoleLPF:
    """一階低通濾波器（用來做高通的 baseline 估計 & 低通）"""
    def __init__(self, fs: int, fc: float):
        a = float(np.exp(-2.0 * np.pi * fc / fs))
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
    """
    0.67–30 Hz（概念）：高通＝移除基線漂移；低通＝抑制肌電高頻；有 SciPy 時再加 50Hz 陷波與中值濾波。
    """
    def __init__(self, fs: int, hp=0.67, lp=30.0, notch_freq=50.0):
        self.fs = fs
        self.hp_lp = OnePoleLPF(fs, hp)   # 用低通近似 baseline，再相減獲得高通效果
        self.lp_lp = OnePoleLPF(fs, lp)
        self.notch_enabled = _HAS_SCIPY
        if self.notch_enabled:
            # 用 iirnotch 簡潔可靠
            b, a = scipy.signal.iirnotch(w0=notch_freq/(fs/2.0), Q=30.0)
            self.notch_b, self.notch_a = b, a
            self.notch_zi = np.zeros(max(len(b), len(a)) - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        # 1) 高通（以低通為 baseline）
        baseline = self.hp_lp.filt_vec(x)
        y = x - baseline

        # 2) 陷波（有 SciPy 才做）
        if self.notch_enabled:
            y, self.notch_zi = scipy.signal.lfilter(
                self.notch_b, self.notch_a, y, zi=self.notch_zi
            )

        # 3) 低通
        y = self.lp_lp.filt_vec(y)

        # 4) 抑制運動偽影（中值濾波，選用）
        if _HAS_SCIPY and len(y) >= 5:
            y = scipy.signal.medfilt(y, kernel_size=5)

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
            diff = np.array([x[0]], dtype=float)
        else:
            diff = np.empty_like(x, dtype=float)
            diff[0] = x[1] - x[0]
            diff[1:] = x[1:] - x[:-1]
        sq = diff * diff

        # ---- 150 ms 移動整流（用環形緩衝取平均）----
        w = self._integ_buf.size
        integ = np.empty_like(sq, dtype=float)
        for i, v in enumerate(sq):
            self._integ_buf[self._integ_idx] = v
            self._integ_idx = (self._integ_idx + 1) % w
            # 這裡要把「當前平均」寫到輸出序列的對應 i 位置，
            # 而不是用環形緩衝的索引去寫 integ！
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

        # --- 設定 ---
        self.fs = int(cfg.get("sampling_rate", 1000))
        ch = cfg.get("analog_channels", [1])
        if not isinstance(ch, list):
            ch = [int(ch)]
        self.analog_channels: List[int] = ch
        self.address: Optional[str] = cfg.get("address") or None

        # --- 讀取 seconds 視窗並推算 buffer ---
        self.seconds_window = int(cfg.get("plot", {}).get("seconds", 10))
        self.buf_len = int(self.seconds_window * self.fs)   # 例如 10s * 1000Hz = 10000
        self.chunk = int(cfg.get("plot", {}).get("chunk", 100))

        # --- 建立 X 軸（0~seconds，不含右端點；固定不變）---
        self._x = np.linspace(0.0, float(self.seconds_window), self.buf_len, endpoint=False)

        # --- 建立 Y 緩衝與曲線 ---
        self._ybuf = np.zeros(self.buf_len, dtype=float)
        self._xpos = 0
        self.curve = self.plot.plot(self._x, self._ybuf, pen=pg.mkPen(width=2))

        # 標籤改成秒
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.showGrid(x=True, y=True)
        # 可選：固定 X 範圍正好 0~seconds
        self.plot.setXRange(0, self.seconds_window, padding=0)

        # --- 濾波 & 偵測 ---
        self.bpf = EnhancedBandpass(self.fs, hp=0.67, lp=30.0, notch_freq=50.0)
        self.det = ECGDetector(self.fs)
        self.warmup_left = int(0.8 * self.fs)  # 暖機 0.8s：先不做 R 偵測
        self.gain = 1.5                        # 顯示增益
        self.alpha = 0.12                      # 穩定 HR 的 EMA 係數
        self.hr_stable: Optional[float] = None

        # --- RR 紀錄（按鈕儲存用） ---
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

    # ------------ 控制 ------------
    def connect_device(self):
        try:
            self.status_bar.showMessage("連線中…")
            self.client.connect()
            self.status_bar.showMessage("BITalino 連線成功", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"連線失敗：{e}")

    def disconnect_device(self):
        try:
            self.stop_stream()
            self.client.close()
            self.status_bar.showMessage("已斷線", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"斷線錯誤：{e}")

    def start_stream(self):
        try:
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

        # 取最後一欄當 ECG（analog_channels=[1] 時通常正好）
        ecg = data[:, -1].astype(float)

        # 濾波 + 增益
        y = self.bpf.process(ecg) * self.gain

        # 畫到滑動窗
        n = len(y)
        p = self._xpos % self.buf_len
        if n >= self.buf_len:
            self._ybuf[:] = y[-self.buf_len:]
            self._xpos += self.buf_len
        else:
            end = p + n
            if end <= self.buf_len:
                self._ybuf[p:end] = y
            else:
                k = self.buf_len - p
                self._ybuf[p:] = y[:k]
                self._ybuf[: (n - k)] = y[k:]
            self._xpos += n
        self.curve.setData(self._x, self._ybuf)

        # 暖機期間不做 R 偵測
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
                self.hr_stable = (1 - self.alpha) * self.hr_stable + self.alpha * hr_inst

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
        Path(fn).write_text("\n".join(f"{v:.1f}" for v in self._rr_accum), encoding="utf-8")
        QtWidgets.QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    def _on_analyze_clicked(self):
        from PyQt6 import QtWidgets
        res = compute_time_domain(self._rr_accum)
        if res is None:
            QtWidgets.QMessageBox.information(None, "HRV 分析", "RR 數量不足，請先擷取 RR。")
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
