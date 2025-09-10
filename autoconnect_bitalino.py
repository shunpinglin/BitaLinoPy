# autoconnect_bitalino.py
# 依序偏好：1) SPP/COM 掃描（優先） 2) BLE 掃描 by 名稱 3)（選擇性）直接使用 preferred_mac
# 需要：pip install bitalino bleak pyserial
"""
Auto-connect helper
- enumerate_candidates(): 掃描並回傳候選清單（COM 優先）
- resolve_address(): 回傳最適合的位址 "COMx" 或 "AA:BB:..."
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional, List, Dict

# --- BLE 掃描 (needs bleak) ---
try:
    from bleak import BleakScanner
    _HAS_BLEAK = True
except Exception:
    _HAS_BLEAK = False

# --- SPP 掃描 (needs pyserial) ---
try:
    from serial.tools import list_ports
    _HAS_PYSERIAL = True
except Exception:
    _HAS_PYSERIAL = False
    list_ports = None


# ---------------------------
# BLE 掃描（同步化封裝）
# ---------------------------
def _ble_discover_blocking(timeout_s: int = 6):
    """
    同步呼叫 BLE 掃描：
    - 無 bleak 或失敗回傳 None
    - 若已有 event loop（某些 UI/測試情境），切到背景執行緒執行，避免 RuntimeError
    """
    if not _HAS_BLEAK:
        return None
    try:
        return asyncio.run(BleakScanner.discover(timeout=timeout_s))
    except RuntimeError:
        # 可能已有 running loop：在臨時執行緒中跑
        out = {}
        def _worker():
            try:
                out["devs"] = asyncio.run(BleakScanner.discover(timeout=timeout_s))
            except Exception as e:
                out["err"] = e
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout_s + 3)
        if "devs" in out:
            return out["devs"]
        return None
    except Exception:
        return None


# ---------------------------
# 工具：清理字串、排序 COM
# ---------------------------
def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _sort_com_key(label: str) -> int:
    """
    嘗試從 'COM12 - ...' / 'COM12' 萃取數字做排序；失敗給大值。
    """
    lbl = label.upper()
    if "COM" in lbl:
        try:
            # 常見 label: "COM12 - USB Serial Device" 或 "COM12"
            after = lbl.split("COM", 1)[1]
            num = ""
            for ch in after:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if num:
                return int(num)
        except Exception:
            pass
    return 9999


# ---------------------------
# 掃描候選裝置
# ---------------------------
def enumerate_candidates(
    name_hints: List[str] = ["BITalino", "bitalino", "plux"],
    timeout_s: int = 6
) -> List[Dict[str, str]]:
    """
    回傳候選裝置清單（字典陣列）：
      [{"type":"COM","address":"COM16","label":"COM: COM16 - Standard Serial over Bluetooth link"},
       {"type":"BLE","address":"AA:BB:...","label":"BLE: BITalino (AA:BB:...)"}, ...]
    規則：
      - COM 優先（多數實務情境）
      - 名稱/描述包含 name_hints、"bitalino"、"standard serial over bluetooth" 皆視為候選
      - 去重：同 (type,address) 只保留一次
    """
    results: List[Dict[str, str]] = []

    # --- COM / SPP（優先） ---
    if _HAS_PYSERIAL and list_ports:
        for p in list_ports.comports():
            desc = _norm(f"{p.description} {p.name} {p.hwid}")
            hits_kw = any(_norm(k) in desc for k in (name_hints or []))
            looks_like_spp = "bitalino" in desc or "standard serial over bluetooth" in desc
            if hits_kw or looks_like_spp:
                results.append({
                    "type": "COM",
                    "address": p.device,  # e.g., "COM16"
                    "label": f"COM: {p.device} - {p.description or 'Serial Port'}"
                })

    # --- BLE（次之） ---
    devs = _ble_discover_blocking(timeout_s=timeout_s)
    if devs:
        for d in devs:
            name = getattr(d, "name", "") or ""
            addr = getattr(d, "address", None)
            if not addr:
                continue
            if any(_norm(k) in _norm(name) for k in (name_hints or [])) or "bitalino" in _norm(name):
                results.append({
                    "type": "BLE",
                    "address": addr,
                    "label": f"BLE: {name or 'Unknown'} ({addr})"
                })

    # 去重（type+address）
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in results:
        key = (r.get("type"), r.get("address"))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # 排序：COM 優先，再依 COM 編號小→大，其次 BLE 依名稱字母序
    def _rank(item: Dict[str, str]):
        if item["type"] == "COM":
            return (0, _sort_com_key(item.get("label", "")))
        else:
            return (1, item.get("label", ""))
    unique.sort(key=_rank)
    return unique


# ---------------------------
# 解析最佳位址
# ---------------------------
def resolve_address(
    preferred_mac: Optional[str] = None,
    preferred_com: Optional[str] = None,
    name_hints: List[str] = ["BITalino", "bitalino", "plux"]
) -> Optional[str]:
    """
    解析最佳位址：
      1) 若指定 preferred_com / preferred_mac，直接回傳
      2) 掃描候選（COM 優先）
      3) 找不到回傳 None
    回傳: "COM16" 或 "AA:BB:CC:DD:EE:FF"
    """
    if preferred_com:
        return preferred_com
    if preferred_mac:
        return preferred_mac

    cands = enumerate_candidates(name_hints=name_hints, timeout_s=6)
    if not cands:
        return None

    # 優先回傳 COM，其次 BLE
    for c in cands:
        if c["type"] == "COM":
            return c["address"]
    return cands[0]["address"]  # 沒有 COM 就回 BLE 的第一個
