# autoconnect_bitalino.py
# 依序嘗試：1) BLE by MAC  2) BLE 掃描 by 名稱  3) SPP/COM 掃描 by 名稱或 VID/PID
# 需要：pip install bitalino bleak pyserial

import time
from typing import Optional, List
from bitalino import BITalino

# --- BLE 掃描 (需要 bleak) ---
try:
    from bleak import BleakScanner
    _HAS_BLEAK = True
except ImportError:
    _HAS_BLEAK = False

# --- SPP 掃描 (需要 pyserial) ---
try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False


def find_ble_mac_by_name(name_keywords: List[str], timeout_s: int = 6) -> Optional[str]:
    """用名稱關鍵字掃 BLE，回傳第一個符合的 MAC"""
    if not _HAS_BLEAK:
        return None
    devices = BleakScanner.discover(timeout=timeout_s)
    for d in devices:
        n = (d.name or "").lower()
        if any(k.lower() in n for k in name_keywords):
            return d.address
    return None


def find_spp_port_by_hint(name_keywords: List[str]) -> Optional[str]:
    """掃描序列埠，依描述尋找 BITalino SPP 的 COMx"""
    if not _HAS_PYSERIAL:
        return None
    for p in list_ports.comports():
        desc = f"{p.description} {p.name} {p.hwid}".lower()
        if any(k.lower() in desc for k in name_keywords) or "bitalino" in desc:
            return p.device
    return None


def connect_bitalino(preferred_mac: Optional[str] = None,
                     name_hints: List[str] = ["BITalino", "bitalino"],
                     sampling_rate: int = 1000,
                     analog_channels: List[int] = [1],
                     retries: int = 3,
                     wait_s: float = 1.5) -> BITalino:
    """自動連線流程：BLE by MAC → BLE 掃描 → SPP/COM 掃描"""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if preferred_mac:
                return BITalino(preferred_mac)

            mac = find_ble_mac_by_name(name_hints, timeout_s=6)
            if mac:
                return BITalino(mac)

            com = find_spp_port_by_hint(name_hints)
            if com:
                return BITalino(com)

            raise RuntimeError("找不到可用的 BITalino (BLE 或 SPP)")
        except Exception as e:
            last_err = e
            time.sleep(wait_s)

    raise RuntimeError(f"BITalino 連線失敗（已重試 {retries} 次）：{last_err}")


# 測試
if __name__ == "__main__":
    PREFERRED_MAC = "bc:33:ac:ab:ad:b4"  # 如果已知 MAC，填在這裡，例如 "bc:33:ac:ab:ad:b4"
    try:
        dev = connect_bitalino(preferred_mac=PREFERRED_MAC)
        print("已連線：", dev)
        dev.start(1000, [1])
        time.sleep(2)
        print("讀取一批資料示例：", dev.read(100))
        dev.stop()
        dev.close()
    except Exception as e:
        print("連線失敗：", e)

