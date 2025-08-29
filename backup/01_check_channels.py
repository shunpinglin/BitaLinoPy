# 01_check_channels.py  -- 掃描 A1..A6，評估哪一軌像 ECG
import tomllib
from pathlib import Path
import numpy as np
from utils.bitalino_helpers import BitalinoClient, split_frame

cfg = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))

# 一次開全通道（0..5 對應 A1..A6）
ach = [0, 1, 2, 3, 4, 5]
client = BitalinoClient(
    mode=cfg["mode"], address=cfg["address"],
    sampling_rate=cfg["sampling_rate"], analog_channels=ach
)

client.connect()
client.start()

# 取一小段資料做統計
frame = client.read_block(1000)  # 約 10 秒 @100Hz / 1 秒 @1000Hz
_, _, analog = split_frame(frame)

client.stop()
client.close()

# 計算每路的均方根(RMS)與帶內能量（粗估）
def band_energy(x, fs):
    # 粗略 0.5–25Hz：先去均值，再用簡單移動平均當低通
    x = x - np.mean(x)
    n = max(1, int(fs/20))  # ~20Hz 低通
    ma = np.convolve(x, np.ones(n)/n, mode="same")
    return float(np.sqrt(np.mean(ma**2)))

fs = int(cfg["sampling_rate"])
r = []
for i in range(analog.shape[1]):
    ch = analog[:, i].astype(float)
    rms = float(np.sqrt(np.mean((ch - np.mean(ch))**2)))
    be = band_energy(ch, fs)
    r.append((i, rms, be))

print("掃描結果 (index, RMS, band_energy~0.5-25Hz)：")
for i, rms, be in r:
    print(f"  ch{i} (A{i+1}): RMS={rms:7.2f}, bandE={be:7.2f}")

best = max(r, key=lambda t: t[2])[0]
print(f"\n建議先在 config.toml 用 analog_channels = [{best}]（A{best+1})")
