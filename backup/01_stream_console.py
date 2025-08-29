# 01_stream_console.py
import tomllib
from pathlib import Path
import numpy as np

from utils.bitalino_helpers import BitalinoClient, split_frame

CFG = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))


client = BitalinoClient(
    mode=CFG["mode"],
    address=CFG["address"],
    sampling_rate=CFG["sampling_rate"],
    analog_channels=CFG["analog_channels"],
)


try:
    print("Connecting...")
    client.connect()
    client.start()
    print("Started. Press Ctrl+C to stop.")

    while True:
        # shape: (N, 1+4+len(analog_channels))
        frame = client.read_block(CFG["block_size"])
        seq, dig, analog = split_frame(frame)
        # 這裡假設只讀一個類比通道（例如 ECG）
        ch0 = analog[:, 0]
        print(
            f"seq[{int(seq[0])}..{int(seq[-1])}] analog0 mean={float(np.mean(ch0)):.1f}")


except KeyboardInterrupt:
    pass
finally:
    client.stop()
    client.close()
print("Stopped.")
