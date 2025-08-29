# 00_detect_ble.py
import asyncio
from bleak import BleakScanner


async def main():
    devices = await BleakScanner.discover(timeout=6.0)

    if not devices:
        print("No BLE devices found.")
        return

    print(f"Found {len(devices)} device(s):")
    for d in devices:
        name = d.name or "(no name)"
        # d.address 就是 MAC（大小寫皆可）
        print(f"{name:30s} {d.address}")

if __name__ == "__main__":
    asyncio.run(main())
