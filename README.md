
# Windows 桌機／筆電通用 venv 啟動小抄

## 一、建立 / 啟用 / 停用

### PowerShell

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
deactivate

> 第一次若被擋：Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

### CMD

py -3.11 -m venv .venv
.\.venv\Scripts\activate
deactivate

### VS Code

- 打開專案資料夾 → Ctrl+` 開終端機
- 若 .vscode/settings.json 已設定，會自動啟用
- 否則手動執行上面 PowerShell/CMD

## 二、常用檢查

python --version
where python
python -c "import PyQt6,pyqtgraph,serial,numpy,pandas; print('✅ imports OK')"

## 三、安裝 / 同步依賴

python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.txt

## 四、日常 SOP（換機器時）

1. 進專案資料夾（不要帶 .venv/）
2. 建立 + 啟用 venv  
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
3. 同步依賴  
   pip install -r requirements.txt

## 五、常見問題速解

- 沒出現 (.venv) → 還沒啟用
- PowerShell 不能執行 Activate.ps1 → Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
- 用到系統 Python 而非 venv → where python 應看到 .venv
- 安裝出現 MSVC / Build Tools 錯誤 → 升級 pip；不行再裝 VS2022 Build Tools
- py 指令不存在 → 改用 python -m venv .venv
- VS Code 抓錯 → Ctrl+Shift+P → Python: Select Interpreter → 選 .venv

your_project/
├─ main.py                          # 應用進入點；載入 UI、建 PlotWidget、建立 ECGController、綁定工具列
├─ config.toml                      # 全域設定（取樣率、通道、濾波、偏好 MAC…）
├─ requirements.txt                 # 依賴套件清單（PyQt6、pyqtgraph、numpy、scipy、bleak、pyserial…）
├─ README.md                        # 專案說明、安裝/執行步驟、教學重點
├─ .gitignore                       # 忽略 .venv、__pycache__、資料輸出等
│
├─ ui/
│  ├─ ui_ecg.py                     # 由 .ui 轉出的 Python 類（Ui_MainWindow）
│  └─ main.ui                       # Qt Designer 原始 UI（教學可視化編修）
│
├─ controllers/
│  └─ ecg_controller.py             # 核心控制器：連線/斷線/啟停串流、緩衝區、即時繪圖、HR/StableHR 計算、RR 存檔、HRV 觸發
│
├─ processing/
│  ├─ filters.py                    # 即時濾波器（sosfilt：帶通＋60Hz陷波，保留 zi 狀態，串流友善）
│  └─ hrv.py (可選)                 # HRV 計算（時域/頻域/MSE…）—之後可拆到這裡
│
├─ devices/
│  └─ bitalino_client.py (可選)     # 與 BITalino 溝通的封裝（連線/讀取/錯誤重試；便於日後支援模擬器/其他裝置）
│
├─ utils/
│  ├─ logging_setup.py              # 統一設定 logging（檔案+主控台；等級/格式）
│  └─ ring_buffer.py (可選)         # 環形緩衝區實作（若目前邏輯分散，可抽離）
│
├─ data/
│  ├─ last_subject.json             # 上次輸入的姓名/年齡/性別（開啟時自動帶入）
│  └─ Results/                      # 輸出資料（RR、PDF、CSV…）
│
├─ autoconnect_bitalino.py (可選)   # 自動掃描/優先 MAC 連線；找不到時回傳友善訊息
└─ scripts/
   ├─ demo_offline_playback.py      # 用離線樣本檔回放資料（教學用，沒有裝置也能跑）
   └─ gen_fake_ecg.py (可選)        # 產生假 ECG + 工頻雜訊/基線漂移，示範濾波前後對比
