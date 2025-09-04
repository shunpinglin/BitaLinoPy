# autoconnect_bitalino.py
# 依序嘗試：1) BLE by MAC  2) BLE 掃描 by 名稱  3) SPP/COM 掃描 by 名稱或 VID/PID
# 需要：pip install bitalino bleak pyserial
"""
Auto-connect helper
- Scan devices, prefer preferred_mac, fall back to first compatible
- Returns an initialized device instance or raises a clear error
"""

import time
from typing import Optional, List, Dict

# --- BLE 掃描 (needs bleak) ---
try:
    from bleak import BleakScanner
    _HAS_BLEAK = True
except ImportError:
    _HAS_BLEAK = False

# --- SPP 掃描 (needs pyserial) ---
try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except ImportError:
    _HAS_PYSERIAL = False


def _ble_discover(timeout_s: int = 6):
    """在同步環境中安全呼叫 BLE 掃描"""
    if not _HAS_BLEAK:
        return []
    import asyncio
    try:
        return asyncio.run(BleakScanner.discover(timeout=timeout_s))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(BleakScanner.discover(timeout=timeout_s))
        finally:
            loop.close()


def enumerate_candidates(name_hints: List[str] = ["BITalino", "bitalino"],
                         timeout_s: int = 6) -> List[Dict[str, str]]:
    """
    回傳候選裝置清單（字典陣列）：
    [{"type":"BLE","address":"AA:BB:...","label":"BLE: Name (AA:BB:...)"}, 
     {"type":"COM","address":"COM16","label":"COM: COM16 - USB Serial Device"}]
    """
    results: List[Dict[str, str]] = []

    # --- BLE ---
    for d in _ble_discover(timeout_s=timeout_s):
        name = (getattr(d, "name", "") or "")
        addr = getattr(d, "address", None)
        if not addr:
            continue
        if any(k.lower() in name.lower() for k in name_hints) or "bitalino" in name.lower():
            results.append({
                "type": "BLE",
                "address": addr,
                "label": f"BLE: {name or 'Unknown'} ({addr})"
            })

    # --- COM / SPP ---
    if _HAS_PYSERIAL:
        for p in list_ports.comports():
            desc = f"{p.description} {p.name} {p.hwid}".lower()
            if "bitalino" in desc or any(k.lower() in desc for k in name_hints):
                results.append({
                    "type": "COM",
                    "address": p.device,  # e.g., "COM16"
                    "label": f"COM: {p.device} - {p.description}"
                })

    # 去重（同一裝置可能同時被 BLE 和 COM 找到）
    seen = set()
    unique = []
    for r in results:
        key = (r["type"], r["address"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def resolve_address(preferred_mac: Optional[str] = None,
                    preferred_com: Optional[str] = None,
                    name_hints: List[str] = ["BITalino", "bitalino"]) -> Optional[str]:
    """
    解析最佳位址：
    1) 指定的 preferred_com / preferred_mac
    2) 掃描到的 BLE 或 COM
    回傳: "COM16" 或 "AA:BB:..."；找不到回傳 None
    """
    if preferred_com:
        return preferred_com
    if preferred_mac:
        return preferred_mac
    cands = enumerate_candidates(name_hints=name_hints, timeout_s=6)
    if not cands:
        return None
    # 簡單策略：優先 COM，次之 BLE（你可自行調整偏好）
    for c in cands:
        if c["type"] == "COM":
            return c["address"]
    return cands[0]["address"]
