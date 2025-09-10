# bitalino_helpers.py
# 用於與 BITalino 設備互動的輔助類別與函式（含自動連線：BLE/COM）
# 需求：pip install bitalino bleak pyserial

from __future__ import annotations

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
except Exception:
    _HAS_BLEAK = False

try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except Exception:
    _HAS_PYSERIAL = False
    list_ports = None


def _find_ble_mac_by_name(name_keywords: List[str], timeout_s: int = 6) -> Optional[str]:
    if not _HAS_BLEAK:
        return None
    # 把 BLE 掃描包裝成同步，避免 event loop 衝突
    import asyncio
    try:
        return asyncio.run(_ble_scan_pick(name_keywords, timeout_s))
    except RuntimeError:
        # 若已有 running loop，換到背景執行緒
        out = {}
        def _worker():
            try:
                out["mac"] = asyncio.run(_ble_scan_pick(name_keywords, timeout_s))
            except Exception as e:
                out["err"] = e
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_s + 3)
        return out.get("mac")

async def _ble_scan_pick(name_keywords: List[str], timeout_s: int) -> Optional[str]:
    devs = await BleakScanner.discover(timeout=timeout_s)
    for d in devs:
        n = (getattr(d, "name", "") or "").lower()
        if any(k.lower() in n for k in name_keywords) or "bitalino" in n:
            return d.address
    return None


def _find_spp_port_by_hint(name_keywords: List[str]) -> Optional[str]:
    if not _HAS_PYSERIAL or not list_ports:
        return None
    try:
        for p in list_ports.comports():
            desc = f"{p.description} {p.name} {p.hwid}".lower()
            looks_like = (
                "bitalino" in desc or
                "standard serial over bluetooth" in desc or
                any(k.lower() in desc for k in name_keywords)
            )
            if looks_like:
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
    自動連線流程（優先：SPP/COM → BLE by MAC/名稱）。
    回傳已連線的 BITalino 物件；失敗會拋出例外。
    """
    if name_hints is None:
        name_hints = ["BITalino", "bitalino", "plux"]

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # 0) 指定地址（可為 COMx 或 BLE MAC）
            if preferred_addr:
                logger.info(f"[auto] 嘗試以指定地址連線：{preferred_addr}")
                return BITalino(preferred_addr)

            # 1) 先找 SPP/COM（Windows 常見）
            com = _find_spp_port_by_hint(name_hints)
            if com:
                logger.info(f"[auto] 以 SPP 找到 COM 連接埠：{com}")
                return BITalino(com)

            # 2) 再找 BLE 名稱
            mac = _find_ble_mac_by_name(name_hints, timeout_s=6)
            if mac:
                logger.info(f"[auto] 以 BLE 找到裝置 MAC：{mac}")
                return BITalino(mac)

            raise RuntimeError("找不到可用的 BITalino (BLE 或 SPP)")
        except Exception as e:
            last_err = e
            logger.warning(f"[auto] 第 {attempt}/{retries} 次連線失敗：{e}")
            if attempt < retries:
                time.sleep(wait_s)

    raise RuntimeError(f"BITalino 連線失敗（已重試 {retries} 次）：{last_err}")


class BitalinoClient:
    """
    高階 BITalino 客戶端：
      - 支援 BLE MAC / 名稱掃描 / SPP COM 掃描 自動連線
      - 背景執行緒連續讀取，透過 callback 丟出資料（numpy.ndarray）
      - on_error: Optional[(Exception)->None]，出錯回呼
    """

    # 提供 on_error 屬性，方便 UI（ECGController）註冊
    on_error: Optional[Callable[[Exception], None]] = None

    def __init__(self):
        self.address: Optional[str] = None
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
        self._chunk_size = 100

    # ---------- 設定與連線 ----------
    def configure(
        self,
        address: Optional[str] = None,
        sampling_rate: int = 1000,
        analog_channels: Optional[List[int]] = None,
        name_hints: Optional[List[str]] = None,
    ):
        self.address = address
        self.sampling_rate = int(sampling_rate)
        if analog_channels is not None:
            # 轉成乾淨的 0..5 整數清單
            self.analog_channels = [int(c) for c in analog_channels if 0 <= int(c) <= 5]
            if not self.analog_channels:
                self.analog_channels = [1]
        if name_hints:
            self.name_hints = list(name_hints)

    def connect(self, retries: int = 3) -> None:
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
        if not self.is_connected or not self.device:
            raise RuntimeError("尚未連線，請先呼叫 connect()")

        if self.is_acquiring:
            logger.info("擷取已經啟動中，略過 start_acquisition()")
            return

        fs = int(self.sampling_rate)
        if fs not in (1, 10, 100, 1000):
            raise ValueError(f"取樣率不支援：{fs}（合法值：1/10/100/1000）")

        chans = [int(c) for c in self.analog_channels] or [1]
        self._chunk_size = int(chunk_size) if chunk_size > 0 else 100
        logger.info(f"開始擷取：fs={fs}, ch={chans}, chunk={self._chunk_size}")

        self._stop_evt.clear()
        try:
            self.device.start(fs, chans)
        except Exception as e:
            logger.exception(f"device.start() 失敗：{e}")
            if hasattr(self, "on_error") and self.on_error:
                try: self.on_error(e)
                except Exception: pass
            raise

        self.is_acquiring = True
        self.acquisition_thread = threading.Thread(
            target=self._acquisition_loop, name="BitalinoAcq", daemon=True
        )
        self.acquisition_thread.start()


    def _acquisition_loop(self):
        assert self.device is not None
        t0 = time.time()
        pushed = 0
        first_shape_logged = False
        try:
            while not self._stop_evt.is_set():
                try:
                    raw = self.device.read(self._chunk_size)
                except Exception as e:
                    logger.exception(f"read() 發生例外：{e}")
                    if hasattr(self, "on_error") and self.on_error:
                        try: self.on_error(e)
                        except Exception: pass
                    break

                if raw is None:
                    time.sleep(0.002)
                    continue

                # 重要：確保是 2D，避免 1D 或 object array 造成後續丟資料
                data = np.atleast_2d(np.asarray(raw))
                if data.size == 0:
                    time.sleep(0.002)
                    continue

                n_samp = int(data.shape[0])
                pushed += n_samp

                if not first_shape_logged:
                    logger.info(f"[acq] 第一包資料 shape={data.shape}, dtype={data.dtype}")
                    first_shape_logged = True

                # 丟給上層（有錯會記 log 並繼續 loop）
                if self.data_callback:
                    try:
                        self.data_callback(data)
                    except Exception as cb_err:
                        logger.exception(f"data_callback 發生例外：{cb_err}")
                        if hasattr(self, "on_error") and self.on_error:
                            try: self.on_error(cb_err)
                            except Exception: pass
                        time.sleep(0.002)

                # 每秒打一個心跳，觀察是否真的有資料
                if time.time() - t0 > 1.0:
                    logger.info(f"[acq] 已送出樣本數（過去 1s）：~{pushed}")
                    pushed = 0
                    t0 = time.time()

                time.sleep(0.0005)

        except Exception as e:
            logger.exception(f"擷取迴圈錯誤：{e}")
            if hasattr(self, "on_error") and self.on_error:
                try: self.on_error(e)
                except Exception: pass
        finally:
            try:
                self.device.stop()
            except Exception as e:
                logger.debug(f"停止裝置時出錯: {e}")
            self.is_acquiring = False
            logger.info("擷取迴圈結束")



    def stop_acquisition(self) -> None:
        if not self.is_acquiring:
            return
        self._stop_evt.set()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=3.0)
        self.is_acquiring = False

    def close(self) -> None:
        """關閉連線（會先停止擷取）；不會在迴圈內自動呼叫。"""
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

    # 選配：設備資訊
    def get_battery_level(self) -> Optional[int]:
        if self.is_connected and self.device:
            try:
                return self.device.battery()
            except Exception as e:
                logger.error(f"取得電池電量失敗: {e}")
        return None

    def get_version(self) -> Optional[str]:
        if self.is_connected and self.device:
            try:
                return self.device.version()
            except Exception as e:
                logger.error(f"取得版本資訊失敗: {e}")
        return None
