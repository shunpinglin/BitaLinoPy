from __future__ import annotations
"""
ECGController —— BITalino 串流→濾波→R 峰→HR/HRV→繪圖（含自動重連）
相依：PyQt6, pyqtgraph, numpy；裝置介面在 utils/bitalino_helpers.py
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
from PyQt6 import QtCore
import pyqtgraph as pg

from utils.bitalino_helpers import BitalinoClient

# 添加 scipy 導入
try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠ SciPy 未安裝，使用簡化濾波功能")


# ===== 一階濾波器（組 0.5–25 Hz 帶通） =====
class OnePoleLPF:
    def __init__(self, fs: int, fc: float):
        self.a = float(np.exp(-2.0 * np.pi * fc / fs))
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


# ===== 增強型濾波器（抗干擾） =====
class EnhancedBandpass:
    """增強型帶通濾波器：消除基線漂移 + 抑制高頻噪聲 + 運動偽影抑制"""
    def __init__(self, fs: int, hp=0.67, lp=30.0, notch_freq=50.0):
        self.fs = fs
        
        # 高通：消除緩慢基線漂移（呼吸影響）
        self.hp_lp = OnePoleLPF(fs, hp)
        
        # 低通：抑制肌電和高頻噪聲
        self.lp_lp = OnePoleLPF(fs, lp)
        
        # 陷波濾波器：消除50/60Hz電源干擾
        self.notch_enabled = SCIPY_AVAILABLE
        if self.notch_enabled:
            self.notch_freq = notch_freq
            self.notch_b, self.notch_a = self._create_notch_filter(notch_freq, 30.0, fs)
            self.notch_zi = np.zeros(max(len(self.notch_b), len(self.notch_a)) - 1)

    def _create_notch_filter(self, freq, q, fs):
        """創建陷波濾波器係數"""
        w0 = 2 * np.pi * freq / fs
        alpha = np.sin(w0) / (2 * q)
        b0 = 1
        b1 = -2 * np.cos(w0)
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        
        return b, a

    def process(self, x: np.ndarray) -> np.ndarray:
        # 1. 消除基線漂移
        baseline = self.hp_lp.filt_vec(x)
        hp_out = x - baseline
        
        # 2. 陷波濾波器消除電源干擾
        if self.notch_enabled:
            hp_out, self.notch_zi = scipy.signal.lfilter(
                self.notch_b, self.notch_a, hp_out, zi=self.notch_zi
            )
        
        # 3. 低通濾波
        filtered = self.lp_lp.filt_vec(hp_out)
        
        # 4. 簡單的運動偽影抑制（中值濾波）
        if SCIPY_AVAILABLE and len(filtered) > 5:
            filtered = scipy.signal.medfilt(filtered, kernel_size=3)
        
        return filtered


# ===== 輕量 R 峰偵測 =====
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
        diff[0] = x[1] - x[0] if x.size > 1 else x[0]
        diff[1:] = x[1:] - x[:-1]
        sq = diff * diff

        # 150 ms 滑動整流
        w = self._integ_buf.size
        integ = np.empty_like(sq)
        for i, v in enumerate(sq):
            self._integ_buf[self._integ_idx] = v
            self._integ_idx = (self._integ_idx + 1) % w
            integ[i] = float(self._integ_buf.mean())

        # 動態門檻（中位數 + 0.8*std）
        self.r_indices.clear(); self.rr_ms.clear()
        med = float(np.median(integ)); std = float(np.std(integ))
        target_thr = med + 0.8 * std
        self._thr = 0.9 * self._thr + 0.1 * target_thr
        refr = int(0.25 * self.fs)  # 250 ms 不應期

        for i in range(1, len(integ) - 1):
            gi = self._global_i + i
            if gi - self._last_peak_i < refr:
                continue
            if integ[i - 1] < integ[i] >= integ[i + 1] and integ[i] > self._thr:
                self.r_indices.append(gi)
                if self._last_peak_i >= 0:
                    rr = (gi - self._last_peak_i) * 1000.0 / self.fs
                    if 300.0 <= rr <= 2000.0:  # 30~200 bpm
                        self.rr_ms.append(rr)
                self._last_peak_i = gi
        self._global_i += len(integ)


# ===== HRV（時域） =====
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


# ===== 主控制器 =====
class ECGController(QtCore.QObject):
    status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        plot_widget: pg.PlotWidget,
        lbl_rt_hr,
        lbl_stable_hr,
        status_bar,
        btn_save_rr,
        btn_analyze,
        cfg: dict,
    ):
        super().__init__()
        self.plot_widget = plot_widget
        self.lbl_rt_hr = lbl_rt_hr
        self.lbl_stable_hr = lbl_stable_hr
        self.status_bar = status_bar
        self.btn_save_rr = btn_save_rr
        self.btn_analyze = btn_analyze

        # ---- 參數 from config ----
        self.fs = int(cfg.get("sampling_rate", 1000))
        self.block_size = int(cfg.get("block_size", 50))
        self.address = cfg.get("address", "")
        # 啟用的 A 通道（例如 [1] 表示只啟用 A2）
        self.sel_channels: List[int] = list(cfg.get("analog_channels", [0])) or [0]
        self.num_sel = len(self.sel_channels)
        # 要顯示 ECG 的那一條：選單中第 0 欄
        self.ecg_sel_idx = 0
        self.ecg_Ai = self.sel_channels[self.ecg_sel_idx]  # A 幾（A1→0, A2→1…）

        # ---- 書寫式繪圖緩衝（8 秒窗）----
        self.win_sec = 8
        self.win = max(100, int(self.win_sec * self.fs))
        self.x = np.arange(self.win) / self.fs
        self.buf = np.full(self.win, np.nan, dtype=float)
        self.head = 0

        # ---- pyqtgraph 曲線（不填色、避免跨 NaN 連線）----
        self.plot_widget.clear()
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(width=2), connect="finite")
        self.curve.setFillLevel(None)
        self.curve.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.plot_widget.addItem(self.curve)

        self.plot_widget.setLabel("bottom", "Time", "s")
        self.plot_widget.setXRange(self.x[0], self.x[-1])
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.enableAutoRange(axis="y", enable=True)

        # ---- 濾波 + 偵測 ----        
        # 替換這行：self.bpf = SimpleBandpass(self.fs, hp=0.5, lp=25.0)
        self.bpf = EnhancedBandpass(self.fs, hp=0.67, lp=30.0, notch_freq=50.0)
        self.det = ECGDetector(self.fs)

        # 顯示/初始化參數
        self.first_block = True                 # 第一次收到資料時做基線初始化
        self.warmup_left = int(0.8 * self.fs)   # 暖機 0.8 秒：先不做 R 偵測
        self.gain = 3.0                         # 顯示增益（可調）
        self.invert = False                     # 若波形上下顛倒可改 True
        self.alpha = 0.1                        # EMA 平滑係數
        self.hr_stable: Optional[float] = None
        self.rr_log_path: Optional[Path] = None

        # 信號質量監測
        self.quality_score = 1.0
        self.last_signal_std = 0.0

        # ---- 自動重連參數 ----
        self.recon_try = 0
        self.recon_enabled = True

        # ---- BITalino client ----
        self.client = BitalinoClient(
            cfg.get("mode", "SERIAL"), self.address, self.fs, self.sel_channels
        )

        # ---- 計時器（由 fs 與 block_size 推得）----
        self.interval_ms = max(1, int(round(1000 * self.block_size / self.fs)))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_once)

        # ---- 綁定 UI 事件 ----
        self.btn_save_rr.clicked.connect(self._on_save_rr)
        self.btn_analyze.clicked.connect(self._on_analyze)

    # ---- 連線 / 開始 / 停止 / 斷線 ----
    def connect_device(self):
        self.client.connect()
        self._set_status(f"✅ 已連線 BITalino（{self.address}），使用通道 A{self.ecg_Ai + 1}")

    def start_stream(self):
        self.recon_try = 0
        self.client.start()
        self.timer.start(self.interval_ms)
        self._set_status(
            f"串流中… fs={self.fs}Hz, 每批={self.block_size} → 更新≈{self.fs / self.block_size:.1f}Hz"
        )

    def stop_stream(self):
        self.timer.stop()
        self.client.stop()
        self._set_status("已停止串流")

    def disconnect_device(self):
        self.timer.stop()
        try:
            self.client.stop()
        except Exception:
            pass
        self.client.close()
        self._set_status("❌ 已斷線")

    # ---- 主更新迴圈 ----
    def _update_once(self):
        try:
            frame = self.client.read_block(self.block_size)
        except Exception as e:
            self._set_status(f"⚠ 讀取錯誤：{e}，準備自動重連…")
            self._auto_reconnect()
            return

        arr = np.asarray(frame)
        if arr.ndim != 2 or arr.size == 0:
            return

        # 取「最後 num_sel 欄」當 analog；其順序 = start() 時的選單順序
        if arr.shape[1] >= self.num_sel:
            analog = arr[:, -self.num_sel:]
        else:
            analog = arr  # 異常 fallback

        # ECG 用選單中第 ecg_sel_idx 欄（通常就是第 0 欄）
        y = analog[:, self.ecg_sel_idx].astype(float)

        # 需要時反相
        if self.invert:
            y = -y

        # 第一次用首樣本做基線初始化（避免一開機大幅衰減）
        if self.first_block and y.size > 0:
            self.bpf.hp_lp.s = float(y[0])  # 高通的低通狀態
            self.bpf.lp_lp.s = 0.0
            self.first_block = False

        # 濾波 + 顯示增益
        y = self.bpf.process(y)
        y *= self.gain

        # 書寫式寫入緩衝
        n = len(y)
        end = self.head + n
        if end <= self.win:
            self.buf[self.head:end] = y
            self.head = end % self.win
        else:
            first = self.win - self.head
            self.buf[self.head:] = y[:first]
            self.buf[: end % self.win] = y[first:]
            self.head = end % self.win
            self.buf[self.head : self.head + 1] = np.nan  # 斷點避免直線回繞

        # 先畫圖（即使暖機也要更新畫面）
        self.curve.setData(self.x, self.buf)

        # 暖機：只畫圖，不做 R 偵測與 HR 計算
        if self.warmup_left > 0:
            self.warmup_left -= n
            return

        # R 峰偵測與 HR 更新
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

            if self.rr_log_path is not None:
                with self.rr_log_path.open("a", encoding="utf-8") as f:
                    for rr in self.det.rr_ms:
                        f.write(f"{rr:.1f}\n")

    def _update_once(self):
        try:
            frame = self.client.read_block(self.block_size)
        except Exception as e:
            self._set_status(f"⚠ 讀取錯誤：{e}，準備自動重連…")
            self._auto_reconnect()
            return

        arr = np.asarray(frame)
        if arr.ndim != 2 or arr.size == 0:
            return

        # 取「最後 num_sel 欄」當 analog；其順序 = start() 時的選單順序
        if arr.shape[1] >= self.num_sel:
            analog = arr[:, -self.num_sel:]
        else:
            analog = arr  # 異常 fallback

        # ECG 用選單中第 ecg_sel_idx 欄（通常就是第 0 欄）
        y = analog[:, self.ecg_sel_idx].astype(float)

        # 需要時反相
        if self.invert:
            y = -y

        # 第一次用首樣本做基線初始化（避免一開機大幅衰減）
        if self.first_block and y.size > 0:
            self.bpf.hp_lp.s = float(y[0])  # 高通的低通狀態
            self.bpf.lp_lp.s = 0.0
            if hasattr(self.bpf, 'notch_zi') and self.bpf.notch_enabled:
                self.bpf.notch_zi = np.zeros_like(self.bpf.notch_zi)
            self.first_block = False

        # 濾波 + 顯示增益
        y_raw = y.copy()  # 保存原始信號用於質量評估
        y = self.bpf.process(y)
        
        # 自適應增益控制
        signal_std = np.std(y) if len(y) > 0 else 1.0
        if signal_std > 0:
            target_std = 0.3  # 目標標準差
            new_gain = target_std / signal_std
            self.gain = 0.9 * self.gain + 0.1 * new_gain
            self.gain = max(0.5, min(3.0, self.gain))  # 限制增益範圍
            
        y *= self.gain
        
        # 信號質量評估
        if len(y) > 10:
            # 計算信噪比
            if SCIPY_AVAILABLE:
                try:
                    smoothed = scipy.signal.medfilt(y, 5)
                    noise = y - smoothed
                    snr = np.std(y) / (np.std(noise) + 1e-10)
                    self.quality_score = min(1.0, max(0.0, snr / 10.0))
                except:
                    # 如果scipy出錯，使用簡化版本
                    current_std = np.std(y)
                    std_change = abs(current_std - self.last_signal_std) / (self.last_signal_std + 1e-10)
                    self.quality_score = max(0.0, 1.0 - min(1.0, std_change))
                    self.last_signal_std = current_std
            else:
                # 簡化版本
                current_std = np.std(y)
                std_change = abs(current_std - self.last_signal_std) / (self.last_signal_std + 1e-10)
                self.quality_score = max(0.0, 1.0 - min(1.0, std_change))
                self.last_signal_std = current_std
            
            if self.quality_score < 0.4:
                self._set_status(f"⚠ 信號質量較低: {self.quality_score:.2f} - 請保持靜止")
            else:
                self._set_status(f"信號質量: {self.quality_score:.2f}")

        # 書寫式寫入緩衝
        n = len(y)
        end = self.head + n
        if end <= self.win:
            self.buf[self.head:end] = y
            self.head = end % self.win
        else:
            first = self.win - self.head
            self.buf[self.head:] = y[:first]
            self.buf[: end % self.win] = y[first:]
            self.head = end % self.win
            self.buf[self.head : self.head + 1] = np.nan  # 斷點避免直線回繞

        # 先畫圖（即使暖機也要更新畫面）
        self.curve.setData(self.x, self.buf)

        # 暖機：只畫圖，不做 R 偵測與 HR 計算
        if self.warmup_left > 0:
            self.warmup_left -= n
            return

        # R 峰偵測與 HR 更新（只在信號質量較好時進行）
        if self.quality_score > 0.3:
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

                if self.rr_log_path is not None:
                    with self.rr_log_path.open("a", encoding="utf-8") as f:
                        for rr in self.det.rr_ms:
                            f.write(f"{rr:.1f}\n")
        else:
            # 信號質量差時清空檢測結果
            self.det.r_indices.clear()
            self.det.rr_ms.clear()
            self.lbl_rt_hr.setText("即時心跳：--")


    # ---- 自動重連 ----
    def _auto_reconnect(self):
        """停止計時器→關閉連線→用遞增等待時間排程重連"""
        if not self.recon_enabled:
            return
        self.timer.stop()
        try:
            try:
                self.client.stop()
            except Exception:
                pass
            self.client.close()
        except Exception:
            pass
        self.recon_try += 1
        delay_s = min(5.0, 0.5 * (2 ** (self.recon_try - 1)))  # 0.5, 1, 2, 4, 5…
        self.lbl_rt_hr.setText("即時心跳：--")
        self.lbl_stable_hr.setText("穩定心跳：-- bpm")
        self._set_status(f"嘗試重新連線（第 {self.recon_try} 次，{delay_s:.1f}s 後）…")
        QtCore.QTimer.singleShot(int(delay_s * 1000), self._reconnect_now)

    def _reconnect_now(self):
        """實際重連；成功就恢復串流，失敗就再排下一次"""
        try:
            self.client.connect()
            self.client.start()
            # 重新初始化顯示與偵測器
            self.first_block = True
            self.warmup_left = int(0.8 * self.fs)
            self.det = ECGDetector(self.fs)
            self.buf[:] = np.nan
            self.head = 0

            self.recon_try = 0
            self.timer.start(self.interval_ms)
            self._set_status("✅ 自動重新連線成功")
        except Exception as e:
            self._set_status(f"❌ 重連失敗：{e}")
            self._auto_reconnect()

    # ---- 儲存 RR（一次匯出目前累積） ----
    def _on_save_rr(self):
        from PyQt6 import QtWidgets

        if len(self.det.rr_ms) == 0:
            QtWidgets.QMessageBox.information(None, "儲存 RR", "目前沒有可儲存的 RR。")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "儲存 RR", f"RR{ts}.txt", "Text Files (*.txt)"
        )
        if not fn:
            return
        Path(fn).write_text("\n".join(f"{v:.1f}" for v in self.det.rr_ms), encoding="utf-8")
        QtWidgets.QMessageBox.information(None, "儲存 RR", f"已儲存：{fn}")

    # ---- HRV（時域） ----
    def _on_analyze(self):
        from PyQt6 import QtWidgets

        res = compute_time_domain(self.det.rr_ms)
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

    # ---- 狀態列 ----
    def _set_status(self, text: str):
        try:
            self.status_bar.showMessage(text)
        except Exception:
            pass
