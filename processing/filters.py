# processing/filters.py
"""
Streaming-safe filters for ECG (real-time)
- Butterworth band-pass (default 0.5–25 Hz)
- Optional powerline notch (50/60 Hz) with Q control
- Uses scipy.signal.sosfilt with internal zi (stateful per chunk)
- Do NOT use filtfilt in streaming contexts
"""

import numpy as np
from typing import Optional, Tuple
try:
    from scipy import signal
except ImportError as e:
    raise RuntimeError("需要 scipy。請先安裝：pip install scipy") from e


class ECGFilterRT:
    """
    即時 ECG 濾波器（串流友善）
    - 先做 50/60Hz 陷波（可選）
    - 再做 Butterworth 帶通（預設 0.5~25 Hz）
    以 sosfilt + 內部狀態實作，不使用 filtfilt（那是離線零相位）
    """

    def __init__(
        self,
        fs: float,
        band: Tuple[float, float] = (0.5, 25.0),
        notch: float = 60.0,
        order: int = 4,
        q: float = 30.0
    ):
        self.fs = float(fs)

        # 帶通設計（Butterworth）
        nyq = self.fs * 0.5
        w = [band[0] / nyq, band[1] / nyq]
        self.sos_bp = signal.butter(order, w, btype="bandpass", output="sos")
        self.zi_bp = signal.sosfilt_zi(self.sos_bp)  # shape: (n_sections, 2)

        # 陷波（可選）
        self.use_notch = notch and notch > 0
        if self.use_notch:
            b, a = signal.iirnotch(w0=notch / nyq, Q=q)
            self.sos_notch = signal.tf2sos(b, a)
            self.zi_notch = signal.sosfilt_zi(self.sos_notch)
        else:
            self.sos_notch = None
            self.zi_notch = None

        self._primed = False  # 初次以首樣值 prime 狀態

    def _prime_state(self, x0: float):
        # 讓初始狀態接近穩態，避免第一批的瞬態
        self.zi_bp = self.zi_bp * x0
        if self.use_notch:
            self.zi_notch = self.zi_notch * x0
        self._primed = True

    def process(self, x: np.ndarray) -> np.ndarray:
        """處理一批資料（1D numpy），會更新內部狀態"""
        if x is None or len(x) == 0:
            return x

        x = np.asarray(x, dtype=float)

        if not self._primed:
            self._prime_state(x[0])

        y = x
        # 先陷波（如果啟用）
        if self.use_notch:
            y, self.zi_notch = signal.sosfilt(
                self.sos_notch, y, zi=self.zi_notch)

        # 再帶通
        y, self.zi_bp = signal.sosfilt(self.sos_bp, y, zi=self.zi_bp)
        return y
