"""
ECG 主視窗（PyQt6 + pyqtgraph）
對應你已經在 Qt Designer 命名好的元件：
- controlWidget：包含 nameEdit、ageEdit、maleRadio、femaleRadio、saveRRButton、analyzeHRVButton、RealTimeHR、StableHR、genderGroup
- chartWidget：放即時 ECG 波形
- statusBar：顯示連線狀態
- ToolBar：加入 連線/斷線/開始/停止/資料來源 等操作

資料來源（三選一）：
- 模擬器（預設可直接跑）
- Serial（Windows 藍牙 SPP 會顯示成 COMx）
- BLE（bleak，預留骨架）

使用方式：
  pip install pyqt6 pyqtgraph pyserial bleak
  python ecg_app_main.py

切換資料來源：
  在視窗右上 ToolBar 的「資料來源」下拉選擇。
  先用「模擬器」檢查 UI 與繪圖，再換 Serial/BLE。

注意：
  - Serial: 修改 DEFAULT_PORT 與鮑率（可能 115200）。
  - BLE: 請填入裝置 Service/Char UUID，或透過掃描功能挑選裝置（已留 TODO）。
"""
from __future__ import annotations

import sys
import math
import time
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QRadioButton, QPushButton, QComboBox, QStatusBar, QMessageBox
)

import pyqtgraph as pg

# === 可選：Serial ===
USE_SERIAL_DEFAULT = False
DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 115200
try:
    import serial
except Exception:
    serial = None

# === 可選：BLE（預留骨架） ===
USE_BLE_DEFAULT = False
try:
    from bleak import BleakClient, BleakScanner  # type: ignore
except Exception:
    BleakClient = None
    BleakScanner = None


@dataclass
class Subject:
    name: str = ""
    age: int = -1
    sex: str = ""

@dataclass
class HRVResults:
    rr_count: int
    mean_rr_ms: float
    sdnn_ms: float
    rmssd_ms: float
    mean_hr_bpm: float


def compute_time_domain(rr_ms: List[int]) -> Optional[HRVResults]:
    if len(rr_ms) < 2:
        return None
    n = len(rr_ms)
    mean_rr = sum(rr_ms) / n
    var = sum((x - mean_rr) ** 2 for x in rr_ms) / (n - 1) if n > 1 else 0
    sdnn = var ** 0.5
    diffs = [(rr_ms[i+1] - rr_ms[i]) for i in range(n - 1)]
    rmssd = (sum(d*d for d in diffs) / (n - 1)) ** 0.5 if n > 1 else 0
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0
    return HRVResults(n, mean_rr, sdnn, rmssd, mean_hr)


# ----------------------------------
# 資料來源介面 & 實作
# ----------------------------------
class DataSource(QObject):
    sample_ready = pyqtSignal(int, float)  # (raw_sample, timestamp_ms)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class SimulatorSource(DataSource):
    """簡單心電波形模擬：200 Hz"""
    def __init__(self, fs: int = 200, parent=None):
        super().__init__(parent)
        self.fs = fs
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.t0 = time.perf_counter()
        self.phase = 0.0
        self.connected.emit()

    def start(self):
        self.timer.start(int(1000 / self.fs))
        self.connected.emit()

    def stop(self):
        self.timer.stop()
        self.disconnected.emit()

    def _tick(self):
        # 基礎正弦 + 簡單尖峰模擬 R 波
        t = time.perf_counter() - self.t0
        base = 512 + 80 * math.sin(2 * math.pi * 1.2 * t)  # 基礎波
        r_spike = 0
        if int(t * 1.2) != int((t - 1 / self.fs) * 1.2):  # 約 1.2 Hz 心率 72 bpm
            r_spike = 400 * math.exp(-((t * 1.2) % 1) * 50)
        sample = int(max(0, min(1023, base + r_spike)))
        self.sample_ready.emit(sample, t * 1000.0)


class SerialWorker(QThread):
    sample_ready = pyqtSignal(int, float)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port = port
        self.baud = baud
        self._running = False
        self._ser = None

    def run(self):
        try:
            if serial is None:
                self.error.emit("未安裝 pyserial，請先 pip install pyserial")
                return
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self._running = True
            self.connected.emit()
            t0 = time.perf_counter()
            while self._running:
                data = self._ser.read(1)
                if not data:
                    continue
                # TODO: 依你的協定解封包；此處假設單一位元組 0-255 為 ECG 值
                val = data[0]
                ts_ms = (time.perf_counter() - t0) * 1000.0
                self.sample_ready.emit(val, ts_ms)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                if self._ser and self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
            self.disconnected.emit()

    def stop(self):
        self._running = False


class SerialSource(DataSource):
    def __init__(self, port: str, baud: int, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud = baud
        self.worker: Optional[SerialWorker] = None

    def start(self):
        self.worker = SerialWorker(self.port, self.baud)
        self.worker.sample_ready.connect(self.sample_ready)
        self.worker.connected.connect(self.connected)
        self.worker.disconnected.connect(self.disconnected)
        self.worker.error.connect(self.error)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None


# 預留 BLESource（骨架）：可依 BITalino UUID 實作 notify 回呼
class BLESource(DataSource):
    def __init__(self, address: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.address = address
        self._running = False
        self._client = None

    def start(self):
        self.error.emit("BLESource 尚未實作：請填入 BITalino 服務/特徵 UUID 與 notify 解析")

    def stop(self):
        pass


# ----------------------------------
# 主視窗
# ----------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG 主視窗（PyQt6）")

        # ==== 取得你在 Designer 命名的元件 ====
        # 若你是用 .ui 轉成 ui_xxx.py，則改成 self.ui = Ui_MainWindow(); self.ui.setupUi(self)
        # 這裡直接在程式動態尋找 objectName（假設你用的是 QMainWindow + centralWidget）
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 模擬：上方控制區（實務上你已經在 .ui 放好，這裡用一個 placeholder）
        self.controlWidget = QWidget(self)
        root.addWidget(self.controlWidget)
        controlLayout = QtWidgets.QGridLayout(self.controlWidget)

        # 對應你的命名
        self.nameEdit = QLineEdit(objectName="nameEdit")
        self.ageEdit = QLineEdit(objectName="ageEdit")
        self.maleRadio = QRadioButton("男", objectName="maleRadio")
        self.femaleRadio = QRadioButton("女", objectName="femaleRadio")
        self.genderGroup = QtWidgets.QButtonGroup(self)
        self.genderGroup.addButton(self.maleRadio)
        self.genderGroup.addButton(self.femaleRadio)
        self.saveRRButton = QPushButton("儲存 RR", objectName="saveRRButton")
        self.analyzeHRVButton = QPushButton("HRV 分析", objectName="analyzeHRVButton")
        self.RealTimeHR = QLabel("即時心跳：-- bpm", objectName="RealTimeHR")
        self.StableHR = QLabel("穩定心跳：-- bpm", objectName="StableHR")

        controlLayout.addWidget(QLabel("姓名"), 0, 0)
        controlLayout.addWidget(self.nameEdit, 0, 1)
        controlLayout.addWidget(QLabel("年齡"), 0, 2)
        controlLayout.addWidget(self.ageEdit, 0, 3)
        controlLayout.addWidget(self.maleRadio, 1, 0)
        controlLayout.addWidget(self.femaleRadio, 1, 1)
        controlLayout.addWidget(self.RealTimeHR, 1, 2)
        controlLayout.addWidget(self.StableHR, 1, 3)
        controlLayout.addWidget(self.saveRRButton, 2, 2)
        controlLayout.addWidget(self.analyzeHRVButton, 2, 3)

        # chart 區
        self.chartWidget = QWidget(self)
        root.addWidget(self.chartWidget, 1)
        chartLayout = QVBoxLayout(self.chartWidget)
        self.plot = pg.PlotWidget()
        chartLayout.addWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen(width=2))
        self.plot.setLabel('left', 'ECG (a.u.)')
        self.plot.setLabel('bottom', 'Sample')
        self.plot.setYRange(0, 1023)

        # 狀態列
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self._set_status("未連線")

        # 工具列
        self._build_toolbar()

        # 緩衝與繪圖
        self.fs = 200  # 預設取樣率（以你的硬體為準）
        self.window = 1000
        self.buffer: Deque[int] = deque([0]*self.window, maxlen=self.window)
        self.ts_prev_ms: Optional[float] = None

        # R 波 & HR 計算
        self.last_r_time_ms: Optional[float] = None
        self.prev_r_time_ms: Optional[float] = None
        self.rr_list: List[int] = []
        self.alpha = 0.1  # 穩定心跳的 EMA 係數
        self.hr_stable: Optional[float] = None

        # 定時重繪（若資料來源主動發 sample，就不一定要）
        self.repaint_timer = QTimer(self)
        self.repaint_timer.timeout.connect(self._repaint)
        self.repaint_timer.start(30)

        # 預設資料來源：模擬器
        self.source: DataSource = SimulatorSource(fs=self.fs)
        self._wire_source(self.source)
        self.source.start()

        # 事件
        self.saveRRButton.clicked.connect(self._on_save_rr)
        self.analyzeHRVButton.clicked.connect(self._on_analyze_hrv)

    # ---------- ToolBar ----------
    def _build_toolbar(self):
        tb = self.addToolBar("工具")
        # 連線/斷線
        self.actConnect = tb.addAction("連線")
        self.actDisconnect = tb.addAction("斷線")
        tb.addSeparator()
        self.actStart = tb.addAction("開始")
        self.actStop = tb.addAction("停止")
        tb.addSeparator()
        # 資料來源
        self.srcCombo = QComboBox()
        self.srcCombo.addItems(["模擬器", "Serial", "BLE"]) 
        tb.addWidget(QtWidgets.QLabel("資料來源："))
        tb.addWidget(self.srcCombo)

        # 事件
        self.actConnect.triggered.connect(self._on_connect)
        self.actDisconnect.triggered.connect(self._on_disconnect)
        self.actStart.triggered.connect(self._on_start)
        self.actStop.triggered.connect(self._on_stop)
        self.srcCombo.currentTextChanged.connect(self._on_source_changed)

    # ---------- 資料來源切換/連線 ----------
    def _on_source_changed(self, text: str):
        self._disconnect_source()
        if text == "模擬器":
            self.source = SimulatorSource(fs=self.fs)
        elif text == "Serial":
            self.source = SerialSource(DEFAULT_PORT, DEFAULT_BAUD)
        else:
            self.source = BLESource()
        self._wire_source(self.source)
        self._set_status(f"來源切換：{text}（未連線）")

    def _wire_source(self, src: DataSource):
        src.sample_ready.connect(self._on_sample)
        src.connected.connect(lambda: self._set_status("✅ 已連線"))
        src.disconnected.connect(lambda: self._set_status("❌ 斷線"))
        src.error.connect(lambda msg: self._set_status(f"⚠ 錯誤：{msg}"))

    def _disconnect_source(self):
        try:
            self.source.stop()
        except Exception:
            pass

    def _on_connect(self):
        self.source.start()

    def _on_disconnect(self):
        self._disconnect_source()

    def _on_start(self):
        # 對 Serial/BLE 來說，start 會開始資料輸入；模擬器已自動跑
        self.source.start()

    def _on_stop(self):
        self.source.stop()

    # ---------- 資料接收/繪圖 ----------
    def _on_sample(self, raw: int, ts_ms: float):
        # 放到視窗緩衝
        self.buffer.append(raw)
        # 簡單高通 + 門檻偵測（示意）
        hr_inst = self._rpeak_and_hr(ts_ms, raw)
        if hr_inst is not None:
            self.RealTimeHR.setText(f"即時心跳：{hr_inst:.0f} bpm")
            # 穩定心跳（EMA）
            if self.hr_stable is None:
                self.hr_stable = hr_inst
            else:
                self.hr_stable = (1 - self.alpha) * self.hr_stable + self.alpha * hr_inst
            self.StableHR.setText(f"穩定心跳：{self.hr_stable:.0f} bpm")

    def _repaint(self):
        self.curve.setData(list(self.buffer))

    # ---------- R 波偵測（示意版） ----------
    def _rpeak_and_hr(self, ts_ms: float, sample: int) -> Optional[float]:
        # 這裡用非常簡化的門檻（請之後替換為你的 R 波偵測）
        TH = 700  # 門檻（視數據調整）
        REFRACTORY = 250  # ms
        hr = None
        if sample > TH:
            if (self.last_r_time_ms is None) or (ts_ms - self.last_r_time_ms > REFRACTORY):
                self.prev_r_time_ms = self.last_r_time_ms
                self.last_r_time_ms = ts_ms
                if self.prev_r_time_ms is not None:
                    rr = int(self.last_r_time_ms - self.prev_r_time_ms)
                    if 300 <= rr <= 2000:
                        self.rr_list.append(rr)
                        hr = 60000.0 / rr
        return hr

    # ---------- 儲存 RR ----------
    def _on_save_rr(self):
        if not self.rr_list:
            QMessageBox.information(self, "儲存 RR", "目前沒有可儲存的 RR 資料。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)")
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write("# RR (ms)\n")
                for rr in self.rr_list:
                    f.write(f"{rr}\n")
            QMessageBox.information(self, "儲存 RR", f"已儲存：{fn}")
        except Exception as e:
            QMessageBox.warning(self, "儲存 RR", f"失敗：{e}")

    # ---------- HRV 分析（時域示範） ----------
    def _on_analyze_hrv(self):
        res = compute_time_domain(self.rr_list)
        if res is None:
            QMessageBox.information(self, "HRV 分析", "RR 數量不足，請先擷取 RR。")
            return
        msg = (
            f"RR 數量：{res.rr_count}\n"
            f"Mean RR：{res.mean_rr_ms:.1f} ms\n"
            f"SDNN：{res.sdnn_ms:.1f} ms\n"
            f"RMSSD：{res.rmssd_ms:.1f} ms\n"
            f"Mean HR：{res.mean_hr_bpm:.1f} bpm\n"
        )
        QMessageBox.information(self, "HRV（時域）", msg)

    def _set_status(self, text: str):
        self.statusBar.showMessage(text)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# utils/bitalino_helpers.py
"""
BITalino 連線與讀取的統一包裝（真機專用）
- 依據 config.toml: mode/address/sampling_rate/analog_channels
- 透過官方 Python 套件 `bitalino` 直接以 MAC 連線（適用 BLE Kit）
- 提供 read_block(n) 介面給上層控制器使用
安裝：
  pip install bitalino numpy
"""
from __future__ import annotations
import time
from typing import List, Tuple

import numpy as np

try:
    from bitalino import BITalino
except Exception as e:  # 延遲導入錯誤到連線時提示
    BITalino = None  # type: ignore


class BitalinoClient:
    def __init__(self, mode: str, address: str, sampling_rate: int, analog_channels: List[int]):
        self.mode = (mode or "BLE").upper()  # 仍保留欄位，但實作走官方 SDK
        self.address = address
        self.fs = int(sampling_rate)
        self.ach = list(analog_channels)
        self.dev: BITalino | None = None

    # --- lifecycle ---
    def connect(self) -> None:
        if BITalino is None:
            raise RuntimeError("未安裝 `bitalino` 套件，請先 pip install bitalino")
        if not self.address:
            raise RuntimeError("config.toml 缺少 address (MAC 位址)")
        # 直接用 MAC 建立連線（BLE Kit 可行）
        self.dev = BITalino(self.address)
        # 設定輸出腳位（通常不需要，保持預設即可）
        # self.dev.setOutput([0, 0])

    def start(self) -> None:
        if not self.dev:
            raise RuntimeError("尚未 connect()")
        # BITalino: start(sampling_rate, analog_channels)
        self.dev.start(self.fs, self.ach)

    def read_block(self, n: int) -> np.ndarray:
        if not self.dev:
            raise RuntimeError("尚未 connect() / start()")
        # 回傳陣列：每列一筆樣本
        return self.dev.read(n)

    def stop(self) -> None:
        if self.dev:
            try:
                self.dev.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self.dev:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None


def split_frame(frame: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    依 BITalino 官方回傳格式：最後 6 欄為 A1..A6 類比訊號。
    回傳：(digital, accel, analog)
    - 為簡化，我們只取 analog（A1..A6），其餘回傳 None。
    """
    arr = np.asarray(frame)
    if arr.ndim != 2 or arr.shape[1] < 6:
        # 防呆：至少要能切到最後 6 欄
        return None, None, arr
    analog = arr[:, -6:]
    return None, None, analog


# controllers/ecg_controller.py
"""
ECGController：把 BITalino 串流 → 濾波 → R 峰偵測 → HR/HRV → 繪圖/儲存
- 將 UI 物件（PlotWidget、RealTimeHR、StableHR、statusBar、按鈕）交進來
- 讀 config.toml 的 fs / block_size / 通道 / 位址
- 真機直連（無模擬）
安裝：
  pip install pyqt6 pyqtgraph numpy bitalino
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
from PyQt6 import QtCore
import pyqtgraph as pg

from utils.bitalino_helpers import BitalinoClient, split_frame


# ===== 一階濾波器（組 0.5–25 Hz 帶通） =====
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
    def __init__(self, fs: int, hp=0.5, lp=25.0):
        self.hp_lp = OnePoleLPF(fs, hp)
        self.lp_lp = OnePoleLPF(fs, lp)
    def process(self, x: np.ndarray) -> np.ndarray:
        baseline = self.hp_lp.filt_vec(x)
        hp = x - baseline
        return self.lp_lp.filt_vec(hp)


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
        refr = int(0.25 * self.fs)
        for i in range(1, len(integ)-1):
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


class ECGController(QtCore.QObject):
    status = QtCore.pyqtSignal(str)

    def __init__(self, plot_widget: pg.PlotWidget, lbl_rt_hr, lbl_stable_hr, status_bar,
                 btn_save_rr, btn_analyze, cfg: dict):
        super().__init__()
        self.plot_widget = plot_widget
        self.lbl_rt_hr = lbl_rt_hr
        self.lbl_stable_hr = lbl_stable_hr
        self.status_bar = status_bar
        self.btn_save_rr = btn_save_rr
        self.btn_analyze = btn_analyze
        self.cfg = cfg

        # 參數
        self.fs = int(cfg.get("sampling_rate", 1000))
        self.block_size = int(cfg.get("block_size", 50))
        self.address = cfg.get("address", "")
        self.analog_channels = list(cfg.get("analog_channels", [1]))

        # 書寫式繪圖設定
        self.win_sec = 8
        self.win = max(100, int(self.win_sec * self.fs))
        self.x = np.arange(self.win) / self.fs
        self.buf = np.full(self.win, np.nan, dtype=float)
        self.head = 0

        # pyqtgraph 曲線
        self.curve = self.plot_widget.plot(pen=pg.mkPen(width=2))
        self.plot_widget.setLabel("bottom", "Time", "s")
        self.plot_widget.setXRange(self.x[0], self.x[-1])
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.enableAutoRange(axis='y', enable=True)

        # 濾波與偵測
        self.bpf = SimpleBandpass(self.fs, hp=0.5, lp=25.0)
        self.det = ECGDetector(self.fs)
        self.alpha = 0.1
        self.hr_stable: Optional[float] = None
        self.rr_log_path: Optional[Path] = None

        # BITalino client
        self.client = BitalinoClient(cfg.get("mode", "BLE"), self.address, self.fs, self.analog_channels)

        # 計時器（依 block_size 自動計算間隔）
        self.interval_ms = max(1, int(round(1000 * self.block_size / self.fs)))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_once)

        # 綁定 UI 事件
        self.btn_save_rr.clicked.connect(self._on_save_rr)
        self.btn_analyze.clicked.connect(self._on_analyze)

    # ---- 連線/開始/停止 ----
    def connect_device(self):
        self.client.connect()
        self._set_status("✅ 已連線 BITalino")

    def start_stream(self):
        self.client.start()
        self.timer.start(self.interval_ms)
        self._set_status(f"串流中… fs={self.fs}Hz, 每批={self.block_size} → 更新≈{self.fs/self.block_size:.1f}Hz")

    def stop_stream(self):
        self.timer.stop()
        self.client.stop()
        self._set_status("已停止串流")

    def disconnect_device(self):
        self.timer.stop()
        try: self.client.stop()
        except Exception: pass
        self.client.close()
        self._set_status("❌ 已斷線")

    # ---- 主回圈 ----
    def _update_once(self):
        try:
            frame = self.client.read_block(self.block_size)
        except Exception as e:
            self._set_status(f"⚠ 讀取錯誤：{e}")
            return
        _, _, analog = split_frame(frame)
        if analog.size == 0:
            return
        y = analog[:, 0].astype(float)
        y = self.bpf.process(y)

        # 書寫式緩衝寫入
        n = len(y); end = self.head + n
        if end <= self.win:
            self.buf[self.head:end] = y
            self.head = end % self.win
        else:
            first = self.win - self.head
            self.buf[self.head:] = y[:first]
            self.buf[:end % self.win] = y[first:]
            self.head = end % self.win
            self.buf[self.head:self.head+1] = np.nan  # 斷線避免回繞直線

        # R 峰與 HR
        self.det.process(y)
        if self.det.rr_ms:
            rr_arr = np.asarray(self.det.rr_ms[-5:], dtype=float)
            hr_inst = 60_000.0 / float(rr_arr.mean())
            self.lbl_rt_hr.setText(f"即時心跳：{hr_inst:.0f} bpm")
            if self.hr_stable is None:
                self.hr_stable = hr_inst
            else:
                self.hr_stable = (1 - self.alpha) * self.hr_stable + self.alpha * hr_inst
            self.lbl_stable_hr.setText(f"穩定心跳：{self.hr_stable:.0f} bpm")
            # RR 追加存檔（若開啟錄製）
            if self.rr_log_path is not None:
                with self.rr_log_path.open("a", encoding="utf-8") as f:
                    for rr in self.det.rr_ms:
                        f.write(f"{rr:.1f}
")

        # 畫圖
        self.curve.setData(self.x, self.buf)

    # ---- RR 存檔 ----
    def _on_save_rr(self):
        from PyQt6 import QtWidgets
        if len(self.det.rr_ms) == 0:
            QtWidgets.QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        dlg = QtWidgets.QFileDialog
        fn, _ = dlg.getSaveFileName(None, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)")
        if not fn:
            return
        Path(fn).write_text("
".join(f"{v:.1f}" for v in self.det.rr_ms), encoding="utf-8")
        QtWidgets.QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    # ---- HRV（時域） ----
    def _on_analyze(self):
        res = compute_time_domain(self.det.rr_ms)
        from PyQt6 import QtWidgets
        if res is None:
            QtWidgets.QMessageBox.information(None, "HRV 分析", "RR 數量不足，請先擷取 RR。")
            return
        msg = (
            f"RR 數量：{res.count}
"
            f"Mean RR：{res.mean_rr:.1f} ms
"
            f"SDNN：{res.sdnn:.1f} ms
"
            f"RMSSD：{res.rmssd:.1f} ms
"
            f"Mean HR：{res.mean_hr:.1f} bpm
"
        )
        QtWidgets.QMessageBox.information(None, "HRV（時域）", msg)

    def _set_status(self, text: str):
        try:
            self.status_bar.showMessage(text)
        except Exception:
            pass


# —— main.py 內的最小掛載示例（請整合到你的主窗體）——
"""
# from PyQt6 import QtWidgets, uic
# import tomllib
# from controllers.ecg_controller import ECGController
# import pyqtgraph as pg
#
# class Main(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = uic.loadUi("your_designed_main.ui", self)
#         cfg = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
#         # 假設你在 chartWidget 裡放了一個 PlotWidget 叫 plotWidget
#         self.controller = ECGController(
#             plot_widget=self.ui.plotWidget,
#             lbl_rt_hr=self.ui.RealTimeHR,
#             lbl_stable_hr=self.ui.StableHR,
#             status_bar=self.statusBar(),
#             btn_save_rr=self.ui.saveRRButton,
#             btn_analyze=self.ui.analyzeHRVButton,
#             cfg=cfg,
#         )
#         # 連線 + 開始串流
#         self.controller.connect_device()
#         self.controller.start_stream()
#
# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#     w = Main(); w.show()
#     app.exec()
"""
