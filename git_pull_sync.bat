@echo off
setlocal ENABLEDELAYEDEXPANSION

where git >nul 2>nul || (echo [ERROR] 未找到 git（請安裝或加入 PATH）。& pause & exit /b 1)

if not exist ".git" (
  echo [ERROR] 這不是 Git 倉庫資料夾。
  echo 初次下載請使用：git clone <repo-url>
  pause
  exit /b 1
)

echo === 一鍵同步（拉回最新） ===

rem 取得目前分支，沒有就預設 main
for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" set "BRANCH=main"
echo 目標分支：%BRANCH%

echo.
echo 1) 取得遠端更新…
git fetch --all --prune || (echo [ERROR] fetch 失敗 & pause & exit /b 1)

rem 確保已設 upstream（origin/BRANCH 存在就綁，否則 fallback 到 origin/main）
git rev-parse --abbrev-ref --symbolic-full-name "%BRANCH%@{u}" >nul 2>nul
if errorlevel 1 (
  git show-ref --verify --quiet refs/remotes/origin/%BRANCH%
  if errorlevel 1 (
    set "BRANCH=main"
    git show-ref --verify --quiet refs/remotes/origin/%BRANCH% || (
      echo [ERROR] 遠端沒有 origin/%BRANCH%，請先建立或切換分支。
      pause & exit /b 1
    )
  )
  git branch --set-upstream-to=origin/%BRANCH% "%BRANCH%" >nul 2>nul
)

echo.
echo 2) 拉回（rebase，自動暫存本地異動）…
git pull --rebase --autostash
if errorlevel 1 (
  echo [WARN] rebase 失敗，嘗試保守模式…
  git pull || (echo [ERROR] pull 仍失敗，請手動解衝突。 & pause & exit /b 1)
)

echo.
echo 3) 更新子模組（若有）…
git submodule update --init --recursive

echo.
echo ✅ 已同步 origin/%BRANCH%
git --no-pager log -1 --oneline
pause
