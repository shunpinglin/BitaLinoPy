# utils/bitalino_helpers.py
import platform
from typing import List, Optional
from bitalino import BITalino

class BitalinoClient:
    def __init__(self, mode: str, address: str, sampling_rate: int, analog_channels: List[int]):
        self.mode = (mode or "SERIAL").upper()
        self.address = address or ""
        self.sampling_rate = int(sampling_rate)
        self.analog_channels = list(analog_channels or [0])
        self.dev: Optional[BITalino] = None

    # ---- 連線 ----
    def connect(self) -> None:
        addr = self.address.strip()
        # Windows 下一律用 COM 埠；避免誤用 MAC
        if platform.system() == "Windows" and not addr.upper().startswith("COM"):
            raise RuntimeError(f"Windows 請在 address 填 COM 埠（例如 COM7），目前是：{addr!r}")
        self.dev = BITalino(addr)

    # ---- 開始串流 ----
    def start(self) -> None:
        if self.dev is None:
            raise RuntimeError("尚未連線：請先呼叫 connect()")

        allowed_fs = {1, 10, 100, 1000}
        fs = int(self.sampling_rate)
        if fs not in allowed_fs:
            raise RuntimeError(f"sampling_rate 必須是 {sorted(allowed_fs)}，目前為 {fs}")

        ach = sorted(set(int(x) for x in self.analog_channels))
        if not ach or any(x < 0 or x > 5 for x in ach):
            raise RuntimeError("analog_channels 必須是 0~5 的整數列表，例如 [0] 或 [1,5]。目前為 " + str(self.analog_channels))

        self.dev.start(fs, ach)

    # ---- 讀取 ----
    def read_block(self, n: int):
        if self.dev is None:
            raise RuntimeError("尚未連線/開始：請先 connect() 與 start()")
        return self.dev.read(int(n))  # 回傳 2D 陣列

    # ---- 停止/關閉 ----
    def stop(self) -> None:
        if self.dev is not None:
            try:
                self.dev.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None
