# 02_stream_plot_pyqt.py  — 準確時間軸 + 右向「書寫式」繪圖 + 0.5–25 Hz 濾波 + HR/RR
import sys
import tomllib
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from utils.bitalino_helpers import BitalinoClient, split_frame


# ===== 一階濾波器（組合成 0.5–25 Hz 帶通） =====
class OnePoleLPF:
    def __init__(self, fs: int, fc: float):
        self.a = float(np.exp(-2.0 * np.pi * fc / fs))
        self.s = 0.0
    def filt_vec(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=float); s = self.s; a = self.a
        for i, v in enumerate(x.astype(float)):
            s = a * s + (1 - a) * v; y[i] = s
        self.s = s; return y

class SimpleBandpass:
    """高通 ~0.5 Hz（移除飄移） + 低通 ~25 Hz（抑制高頻/肌電）"""
    def __init__(self, fs: int, hp=0.5, lp=25.0):
        self.hp_lp = OnePoleLPF(fs, hp)
        self.lp_lp = OnePoleLPF(fs, lp)
    def process(self, x: np.ndarray) -> np.ndarray:
        baseline = self.hp_lp.filt_vec(x)   # 長期低通 → 基線
        hp = x - baseline                   # 高通
        return self.lp_lp.filt_vec(hp)      # 再低通


# ===== 輕量 R 峰偵測（微分→平方→移動整流→門檻+不應期） =====
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
        diff = np.empty_like(x)
        diff[0] = x[1] - x[0] if len(x) > 1 else x[0]
        diff[1:] = x[1:] - x[:-1]
        sq = diff * diff

        w = self._integ_buf.size
        integ = np.empty_like(sq)
        for i, v in enumerate(sq):
            self._integ_buf[self._integ_idx] = v
            self._integ_idx = (self._integ_idx + 1) % w
            integ[i] = float(self._integ_buf.mean())

        self.r_indices.clear(); self.rr_ms.clear()
        med = float(np.median(integ)); std = float(np.std(integ))
        target_thr = med + 0.8 * std
        self._thr = 0.9 * self._thr + 0.1 * target_thr
        refr = int(0.25 * self.fs)  # 250 ms 不應期

        for i in range(1, len(integ)-1):
            gi = self._global_i + i
            if gi - self._last_peak_i < refr: continue
            if integ[i-1] < integ[i] >= integ[i+1] and integ[i] > self._thr:
                self.r_indices.append(gi)
                if self._last_peak_i >= 0:
                    rr = (gi - self._last_peak_i) * 1000.0 / self.fs
                    if 300.0 <= rr <= 2000.0:  # 30~200 bpm
                        self.rr_ms.append(rr)
                self._last_peak_i = gi
        self._global_i += len(integ)


# ===== GUI + 串流 =====
class LivePlot(QtWidgets.QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("BITalino Live Plot"); self.resize(1100, 600)

        # Plot
        self.plot = pg.PlotWidget(); self.setCentralWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        self.plot.showGrid(x=True, y=True)

        # 參數
        self.cfg = cfg
        self.fs = int(cfg["sampling_rate"])              # 1/10/100/1000
        self.win_sec = 8                                  # 視窗寬度（秒）
        self.win = max(100, int(self.win_sec * self.fs))
        self.x = np.arange(self.win) / self.fs
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.setXRange(self.x[0], self.x[-1])

        # 書寫式緩衝：左→右畫，滿了回開頭；用 NaN 斷線避免回繞直線
        self.buf = np.full(self.win, np.nan, dtype=float)
        self.head = 0

        # 顯示調校
        self.gain = 2.0
        self.invert = False
        self.plot.enableAutoRange(axis='y', enable=True)

        # HR 顯示
        self.hr_label = QtWidgets.QLabel("HR: -- bpm")
        self.statusBar().addPermanentWidget(self.hr_label)

        # RR 錄製
        self.record_action = QtGui.QAction("Record RR", self)
        self.record_action.setCheckable(True)
        self.record_action.toggled.connect(self._toggle_record)
        tb = self.addToolBar("Controls"); tb.addAction(self.record_action)
        self._rr_fp: Optional[Path] = None

        # 濾波器 + 偵測器
        self.bpf = SimpleBandpass(self.fs, hp=0.5, lp=25.0)
        self.det = ECGDetector(self.fs)

        # BITalino
        self.client = BitalinoClient(
            mode=cfg["mode"], address=cfg["address"],
            sampling_rate=self.fs, analog_channels=cfg["analog_channels"]
        )
        self.client.connect(); self.client.start()

        # ==== 正確的定時器節奏（讓時間軸準）====
        self.target_update_hz = 50                      # 想要的更新頻率
        self.block_n = max(1, int(round(self.fs / self.target_update_hz)))   # 每批讀點
        self.interval_ms = max(1, int(round(1000 * self.block_n / self.fs))) # 觸發間隔
        self.actual_update_hz = self.fs / self.block_n                         # 實際更新頻率
        self.statusBar().showMessage(
            f"fs={self.fs}Hz  update≈{self.actual_update_hz:.1f}Hz  block={self.block_n}", 3000
        )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(self.interval_ms)

    # -- RR 錄製 --
    def _toggle_record(self, enabled: bool):
        if enabled:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._rr_fp = Path(f"RR{ts}.txt")
            self.statusBar().showMessage(f"Recording RR → {self._rr_fp.name}", 3000)
        else:
            self._rr_fp = None
            self.statusBar().showMessage("Recording stopped", 2000)

    # -- 主更新 --
    def update_data(self):
        try:
            frame = self.client.read_block(self.block_n)   # 小批量高頻率讀
        except Exception:
            return
        _, _, analog = split_frame(frame)

        y = analog[:, 0].astype(float)
        y = self.bpf.process(y)
        if self.invert: y = -y
        y *= self.gain

        # 書寫式寫入（左→右）；回繞時在斷點放 NaN
        n = len(y); end = self.head + n
        if end <= self.win:
            self.buf[self.head:end] = y
            self.head = end % self.win
        else:
            first = self.win - self.head
            self.buf[self.head:] = y[:first]
            self.buf[:end % self.win] = y[first:]
            self.head = end % self.win
            self.buf[self.head:self.head+1] = np.nan   # 斷線

        # HR/RR
        self.det.process(y)
        hr_txt = "HR: -- bpm"
        if self.det.rr_ms:
            rr_arr = np.array(self.det.rr_ms[-5:], dtype=float)
            hr = 60_000.0 / float(np.mean(rr_arr))
            hr_txt = f"HR: {hr:5.1f} bpm"
            if self._rr_fp is not None:
                with self._rr_fp.open("a", encoding="utf-8") as f:
                    for rr in self.det.rr_ms:
                        f.write(f"{rr:.1f}\n")
        self.hr_label.setText(hr_txt)

        # 顯示
        self.curve.setData(self.x, self.buf)

    def closeEvent(self, e):
        self.timer.stop()
        try: self.client.stop()
        finally: self.client.close()
        return super().closeEvent(e)


if __name__ == "__main__":
    cfg = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
    app = QtWidgets.QApplication(sys.argv)
    w = LivePlot(cfg); w.show()
    sys.exit(app.exec())







