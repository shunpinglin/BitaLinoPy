
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