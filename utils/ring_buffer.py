# ring_buffer.py
"""
BITalino device wrapper
- Discover/auto-connect (BLE/serial), set sampling rate/channels
- Read chunks (numpy array), simple retries & error mapping
- Single responsibility: IO only (no plotting/processing)
"""
