# bitalino_helpers.py
# 用於與 BITalino 設備互動的輔助類別與函式（含自動連線：BLE/COM）
# 需求：pip install bitalino bleak pyserial

import time
import logging
import threading
from typing import List, Optional, Callable
import numpy as np
from bitalino import BITalino

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BitalinoClient")


# ===== 選配：BLE 與 SPP 掃描能力 =====
try:
    from bleak import BleakScanner
    _HAS_BLEAK = True
except ImportError:
    _HAS_BLEAK = False

try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False


def _find_ble_mac_by_name(name_keywords: List[str], timeout_s: int = 6) -> Optional[str]:
    """用名稱關鍵字掃 BLE，回傳第一個符合的 MAC；無 bleak 或找不到則回傳 None。"""
    if not _HAS_BLEAK:
        return None
    try:
        devices = BleakScanner.discover(timeout=timeout_s)
        for d in devices:
            n = (d.name or "").lower()
            if any(k.lower() in n for k in name_keywords):
                return d.address
    except Exception as e:
        logger.warning(f"BLE 掃描失敗: {e}")
    return None


def _find_spp_port_by_hint(name_keywords: List[str]) -> Optional[str]:
    """掃描序列埠，依描述/名稱尋找 BITalino SPP 對應的 COMx；無 pyserial 或找不到則回傳 None。"""
    if not _HAS_PYSERIAL:
        return None
    try:    
        for p in list_ports.comports():
            desc = f"{p.description} {p.name} {p.hwid}".lower()
            if any(k.lower() in desc for k in name_keywords) or "bitalino" in desc:
                return p.device  # e.g., 'COM7'
    except Exception as e:
        logger.warning(f"COM 端口掃描失敗: {e}")
    return None


def _auto_connect_bitalino(
    preferred_addr: Optional[str] = None,
    name_hints: Optional[List[str]] = None,
    retries: int = 3,
    wait_s: float = 1.5,
) -> BITalino:
    """
    自動連線流程（優先：BLE by MAC/地址 → BLE 掃描 by 名稱 → SPP/COM 掃描）。
    回傳已連線的 BITalino 物件；失敗會拋出例外。
    """
    if name_hints is None:
        name_hints = ["BITalino", "bitalino", "plux"]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # 1) 指定地址：可能是 BLE MAC（XX:XX:...）或 Windows 的 'COMx'
            if preferred_addr:
                logger.info(f"[auto] 嘗試以指定地址連線：{preferred_addr}")
                return BITalino(preferred_addr)

            # 2) BLE 掃描找名稱
            mac = _find_ble_mac_by_name(name_hints, timeout_s=6)
            if mac:
                logger.info(f"[auto] 以 BLE 找到裝置 MAC：{mac}")
                return BITalino(mac)

            # 3) SPP 掃描找 COM Port
            logger.info("[auto] 掃描 COM 端口...")
            com = _find_spp_port_by_hint(name_hints)
            if com:
                logger.info(f"[auto] 以 SPP 找到 COM 連接埠：{com}")
                return BITalino(com)

            raise RuntimeError("找不到可用的 BITalino (BLE 或 SPP)")
        except Exception as e:
            last_err = e
            logger.warning(f"[auto] 第 {attempt}/{retries} 次連線失敗：{e}")
            if attempt < retries:
                time.sleep(wait_s)

    raise RuntimeError(f"BITalino 連線失敗（已重試 {retries} 次）：{last_err}")


# ===== 小工具 =====
class _safe_suppress:
    """用法：with _safe_suppress(): ... —— 靜默忽略例外（logging.debug 紀錄即可）"""
    def enter(self):
        return self
    def exit(self, exc_type, exc, tb):
        if exc:
            logger.debug(f"suppress: {exc}")
        return True # 抑制例外

class BitalinoClient:
    """
    高階 BITalino 客戶端：
      - 支援 BLE MAC / 名稱掃描 / SPP COM 掃描 自動連線
      - 背景執行緒連續讀取，透過 callback 丟出資料（numpy ndarray）
    """

    def __init__(self):
        self.address: Optional[str] = None          # 可填 MAC 或 'COMx'；不填則自動掃描
        self.name_hints: List[str] = ["BITalino", "bitalino", "plux"]

        self.sampling_rate: int = 1000
        self.analog_channels: List[int] = [1]

        self.device: Optional[BITalino] = None
        self.is_connected: bool = False
        self.is_acquiring: bool = False

        # data_callback: (np.ndarray) -> None
        self.data_callback: Optional[Callable[[np.ndarray], None]] = None

        self.acquisition_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._chunk_size = 100  # 每次 read 的筆數（你可以視需要調整）

    # ---------- 設定與連線 ----------

    def configure(
        self,
        address: Optional[str] = None,
        sampling_rate: int = 1000,
        analog_channels: List[int] = None,
        name_hints: Optional[List[str]] = None,
    ):
        """
        設定裝置參數。
        address: 可為 BLE MAC（'XX:XX:...'}）或 'COMx'；None 代表交給自動掃描
        """
        self.address = address
        self.sampling_rate = sampling_rate
        if analog_channels is not None:
            self.analog_channels = list(analog_channels)
        if name_hints:
            self.name_hints = list(name_hints)

    def connect(self, retries: int = 3) -> None:
        """建立連線（自動 BLE/COM 掃描）。"""
        if self.is_connected and self.device:
            logger.info("已連線，略過 connect()")
            return

        logger.info("嘗試連線 BITalino …")
        dev = _auto_connect_bitalino(
            preferred_addr=self.address,
            name_hints=self.name_hints,
            retries=retries,
            wait_s=1.5,
        )
        self.device = dev
        self.is_connected = True
        logger.info("BITalino 連線成功")

    # ---------- 資料擷取 ----------

    def start_acquisition(self, chunk_size: int = 100) -> None:
        """
        開始擷取資料；資料會在背景執行緒連續 read()，並呼叫 data_callback。
        """
        if not self.is_connected or not self.device:
            raise RuntimeError("尚未連線，請先呼叫 connect()")

        if self.is_acquiring:
            logger.info("擷取已經啟動中，略過 start_acquisition()")
            return

        self._chunk_size = int(chunk_size) if chunk_size > 0 else 100
        logger.info(f"開始擷取：fs={self.sampling_rate}, ch={self.analog_channels}, chunk={self._chunk_size}")

        # 啟動裝置與背景執行緒
        self.device.start(self.sampling_rate, self.analog_channels)
        self._stop_evt.clear()
        self.is_acquiring = True

        self.acquisition_thread = threading.Thread(
            target=self._acquisition_loop, name="BitalinoAcq", daemon=True
        )
        self.acquisition_thread.start()

    def _acquisition_loop(self):
        """背景擷取迴圈；將資料轉成 numpy.ndarray 傳給 callback。"""
        assert self.device is not None
        try:
            while not self._stop_evt.is_set():
                # bitalino.read(n) 回傳 shape 約為 (n, 6+) 的列表/ndarray（依韌體/通道而異）
                raw = self.device.read(self._chunk_size)
                data = np.array(raw)  # 轉成 ndarray，方便後處理/繪圖

                # 丟給 callback（若有）
                if self.data_callback:
                    try:
                        self.data_callback(data)
                    except Exception as cb_err:
                        logger.exception(f"data_callback 發生例外：{cb_err}")

        except Exception as e:
            logger.exception(f"擷取迴圈錯誤：{e}")
        finally:
            # 走到這裡代表要停；確保裝置停止
            try:
                self.device.stop()
            except Exception as e:
                logger.debug(f"停止裝置時出錯: {e}")
            self.is_acquiring = False
            logger.info("擷取迴圈結束")

    def stop_acquisition(self) -> None:
        """停止擷取資料與背景執行緒。"""
        if not self.is_acquiring:
            return
        self._stop_evt.set()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=3.0)
        # device.stop() 已在 loop 的 finally 內保險呼叫
        self.is_acquiring = False

    def close(self) -> None:
        """關閉連線（會先停止擷取）。"""
        try:
            self.stop_acquisition()
        except Exception as e:
            logger.debug(f"停止擷取時出錯: {e}")
        finally:
            if self.device:
               try:
                    self.device.close()
               except Exception as e:
                    logger.debug(f"關閉裝置時出錯: {e}")
               self.device = None

            self.is_connected = False
            logger.info("BITalino 連線已關閉")

    def get_battery_level(self) -> Optional[int]:
        """取得電池電量（mV）"""
        if self.is_connected and self.device:
            try:
                return self.device.battery()
            except Exception as e:
                logger.error(f"取得電池電量失敗: {e}")
        return None

    def get_version(self) -> Optional[str]:
        """取得設備版本資訊"""
        if self.is_connected and self.device:
            try:
                return self.device.version()
            except Exception as e:
                logger.error(f"取得版本資訊失敗: {e}")
        return None