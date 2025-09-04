@echo off
chcp 65001 >nul
title Git Menu (safe config)

:: 檢查 git
where git >nul 2>nul || (echo [ERROR] Git 未安裝或未加入 PATH。& pause & exit /b 1)

:: 檢查是否在 git 倉庫內
git rev-parse --is-inside-work-tree >nul 2>nul || (
  echo 初始化 git 倉庫...
  git init || (echo [ERROR] git init 失敗。& pause & exit /b 1)
)

:: 讀取/設定分支名
set "BRANCH="
set /p BRANCH=Enter branch name (default main): 
if "%BRANCH%"=="" set "BRANCH=main"

:: 將目前分支改名為指定分支（常見：main）
git branch -M "%BRANCH%" >nul 2>nul

:: 顯示當前分支
for /f "delims=" %%b in ('git branch --show-current') do set "CURBR=%%b"
echo.
echo 現在分支: %CURBR%

:: 讓使用者輸入 commit 訊息（可留空）
set "MSG="
set /p MSG=Commit message (可留空): 
if "%MSG%"=="" set "MSG=update"

echo.
echo [1/3] add ..
git add -A

echo [2/3] commit ..
git commit -m "%MSG%" || echo (沒有可提交的變更)

:: 確保遠端 origin
git remote get-url origin >nul 2>nul
if errorlevel 1 (
  echo.
  echo 尚未設定遠端 origin。
  set "URL="
  set /p URL=請貼上遠端倉庫 URL (例如 https://github.com/you/repo.git): 
  if "%URL%"=="" (echo [ERROR] 未提供遠端 URL。& pause & exit /b 1)
  git remote add origin "%URL%" || (echo [ERROR] 設定 origin 失敗。& pause & exit /b 1)
) else (
  for /f "delims=" %%u in ('git remote get-url origin') do set "URL=%%u"
  echo 使用遠端: %URL%
)

:: ====== 關鍵：避免重複寫入 config ======
call :set_tracking origin "%BRANCH%"

:: 如果沒有 upstream，就用 -u；有的話一般 push
git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
  echo [3/3] push（建立追蹤）..
  git push -u origin "%BRANCH%"
) else (
  echo [3/3] push ..
  git push
)

echo.
git status -sb
echo 完成！
pause
exit /b 0

:: ---------------------------
:: 以「覆寫」方式設定追蹤分支，先清乾淨再寫一次
:: 用法：call :set_tracking origin branchName
:: ---------------------------
:set_tracking
set "REMOTE=%~1"
set "BR=%~2"

:: 先移除所有既有值（避免 has multiple values）
git config --unset-all branch.%BR%.remote >nul 2>nul
git config --unset-all branch.%BR%.merge  >nul 2>nul

:: 再正確寫回一次
git config branch.%BR%.remote %REMOTE%
git config branch.%BR%.merge  refs/heads/%BR%

:: （保險）驗證
echo.
echo Tracking config for [%BR%]:
git config --get-all branch.%BR%.remote
git config --get-all branch.%BR%.merge
echo.
exit /b 0




