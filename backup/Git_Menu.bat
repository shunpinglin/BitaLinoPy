@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title Git Menu Tool

:: ===== Folder status check =====
echo.
echo ========================================
echo          FOLDER STATUS CHECK
echo ========================================
echo Working Directory: %cd%
echo.

:: Check Git status
set "GIT_STATUS=NOT_INITIALIZED"
set "GIT_BRANCH="
set "GIT_REMOTE=NOT_SET"

if exist ".git" (
    set "GIT_STATUS=INITIALIZED"
    
    :: Get current branch (single token)
    for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "GIT_BRANCH=%%b"
    if defined GIT_BRANCH (
        echo Current Branch: !GIT_BRANCH!
    ) else (
        echo Current Branch: NO_COMMITS_YET
    )
    
    :: Check remote settings
    git remote get-url origin >nul 2>nul
    if not errorlevel 1 (
        set "GIT_REMOTE=SET"
        echo Remote Repository: 
        git remote -v
    ) else (
        echo Remote Repository: NOT_SET
    )
) else (
    echo Git Status: %GIT_STATUS%
    echo.
    echo WARNING: WARNING: WARNING
    echo Current directory is not a Git repository!
    echo Please use 'Initialize' function first
    echo.
)

echo ========================================
echo.

:: Safety confirmation (if not Git repository)
if not exist ".git" (
    set /p "CONFIRM=Continue in non-Git directory? (y/N): "
    if /i not "%CONFIRM%"=="y" (
        echo Operation cancelled
        timeout /t 2 >nul
        exit /b 0
    )
    echo.
)

:: Check if Git exists
where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git not found, please install Git first
    echo Download: https://git-scm.com/download/win
    pause
    exit /b 1
)

:MAIN_MENU
cls
echo.
echo ==========================================================
echo                       GIT MENU TOOL 2025.09.01
echo ==========================================================
echo Folder: %cd%
if exist ".git" (
    for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "CURBR=%%b"
    if defined CURBR (echo Branch: !CURBR!) else (echo Branch: NO_COMMITS_YET)
)
echo ==========================================================
echo.
echo  1) Initialize Repository  (git init) / set main
echo  2) User Configuration (user.name / user.email)
echo  3) Create / edit .gitignore
echo  4) Set or change remote "origin" / Remote URL
echo  5) Show status (git status)
echo  6) Add and commit (git add . ; git commit)
echo  7) Push to remote (git push)
echo  8) Pull from remote (git pull)
echo  9) Create and switch to new branch (git checkout -b)
echo 10) Switch to existing branch (git checkout)
echo 11) Merge into current branch (git merge)
echo 12) Show concise history: git log --oneline --graph --all
echo 13) Soft reset to previous commit (keep file changes)
echo 14) HARD reset to previous commit (discard changes)
echo 15) Enable Git Credential Manager (GUI login on push)
echo 16) List Branches
echo 17) Delete Branch
echo  0) Exit Program
echo.
echo ==========================================================
set /p "CHOICE=Enter choice 0-17: "

if "%CHOICE%"=="1" goto INIT
if "%CHOICE%"=="2" goto CONFIG_USER
if "%CHOICE%"=="3" goto GITIGNORE
if "%CHOICE%"=="4" goto SET_REMOTE
if "%CHOICE%"=="5" goto STATUS
if "%CHOICE%"=="6" goto ADD_COMMIT
if "%CHOICE%"=="7" goto PUSH
if "%CHOICE%"=="8" goto PULL
if "%CHOICE%"=="9" goto NEW_BRANCH
if "%CHOICE%"=="10" goto SWITCH_BRANCH
if "%CHOICE%"=="11" goto MERGE_BRANCH
if "%CHOICE%"=="12" goto LOG_HISTORY
if "%CHOICE%"=="13" goto RESET_SOFT
if "%CHOICE%"=="14" goto RESET_HARD
if "%CHOICE%"=="15" goto CREDENTIAL_MANAGER
if "%CHOICE%"=="16" goto SHOW_BRANCHES
if "%CHOICE%"=="17" goto DELETE_BRANCH
if "%CHOICE%"=="0" goto EXIT

echo Invalid choice, please try again
pause >nul
goto MAIN_MENU

:: ============ FUNCTION IMPLEMENTATION ============

:INIT
echo.
if exist ".git" (
    echo Repository already exists
    echo Current status:
    git status -sb
) else (
    git init
    if errorlevel 1 (
        echo Initialization failed
    ) else (
        echo Git repository created successfully
        rem Set default branch name globally (once)
        git config --global init.defaultBranch main >nul 2>nul
        git branch -M main 2>nul || echo Make first commit to name branch
    )
)
pause
goto MAIN_MENU

:CONFIG_USER
echo.
echo Current user settings:
git config user.name || echo Not set
git config user.email || echo Not set
echo.
set /p "USERNAME=Enter username: "
if not "%USERNAME%"=="" (
    git config user.name "%USERNAME%"
    if errorlevel 1 (
        echo Username setting failed
    ) else (
        echo Username saved
    )
)

echo.
set /p "EMAIL=Enter email: "
if not "%EMAIL%"=="" (
    git config user.email "%EMAIL%"
    if errorlevel 1 (
        echo Email setting failed
    ) else (
        echo Email saved
    )
)

echo.
echo Updated settings:
git config user.name || echo Not set
git config user.email || echo Not set
pause
goto MAIN_MENU

:GITIGNORE
echo.
if not exist ".gitignore" (
    echo Creating default gitignore...
    (
        echo # Python
        echo __pycache__/
        echo *.pyc
        echo *.pyo
        echo .venv/
        echo venv/
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo *.swp
        echo.
        echo # OS
        echo .DS_Store
        echo Thumbs.db
        echo.
        echo # Build
        echo build/
        echo dist/
        echo node_modules/
    ) > .gitignore
    echo gitignore file created
) else (
    echo Opening gitignore for editing...
    notepad .gitignore || (
        echo Please edit .gitignore manually
    )
)
pause
goto MAIN_MENU

:SET_REMOTE
echo.
echo Current remote:
git remote -v
echo.
set /p "REMOTE_URL=Enter remote URL: "
if not "%REMOTE_URL%"=="" (
    git remote get-url origin >nul 2>nul
    if not errorlevel 1 (
        git remote set-url origin "%REMOTE_URL%"
    ) else (
        git remote add origin "%REMOTE_URL%"
    )
    if errorlevel 1 (
        echo Remote setup failed
    ) else (
        echo Remote URL configured
        git remote -v
    )
)
pause
goto MAIN_MENU

:STATUS
echo.
git status -sb
pause
goto MAIN_MENU

:ADD_COMMIT
echo.
set /p "COMMIT_MESSAGE=Enter commit message: "
if "%COMMIT_MESSAGE%"=="" (
    set "COMMIT_MESSAGE=Code update"
)
git add .
if errorlevel 1 (
    echo File add failed
    pause
    goto MAIN_MENU
)
git commit -m "%COMMIT_MESSAGE%"
if errorlevel 1 (
    echo Commit failed, maybe no changes
)
pause
goto MAIN_MENU

:PUSH
echo.
set "BRANCH="
set /p "BRANCH=Enter branch name (default: current or main): "
if "%BRANCH%"=="" (
    for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "BRANCH=%%b"
    if "%BRANCH%"=="" set "BRANCH=main"
)
echo Target branch: %BRANCH%

:: Ensure origin exists
call :ensure_origin || (
  echo [ERROR] 遠端 origin 尚未設定且設定失敗
  pause
  goto MAIN_MENU
)

:: Avoid duplicate config: reset then set tracking
call :set_tracking "%BRANCH%"

echo.
echo 將推送到：origin/%BRANCH%
:: Check upstream for the SPECIFIED branch (not only current)
git rev-parse --abbrev-ref --symbolic-full-name "%BRANCH%@{u}" >nul 2>nul
if errorlevel 1 (
  echo (首次推送，建立追蹤)
  git push -u origin "%BRANCH%"
) else (
  git push origin "%BRANCH%"
)
if errorlevel 1 (
  echo Push failed
) else (
  echo Push OK
)
pause
goto MAIN_MENU

:PULL
echo.
set "BRANCH="
set /p "BRANCH=Enter branch name (default: current): "
if "%BRANCH%"=="" (
    for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "BRANCH=%%b"
)
if "%BRANCH%"=="" set "BRANCH=main"
echo Pulling from origin/%BRANCH%...
git pull --rebase origin %BRANCH%
if errorlevel 1 (
    echo Pull failed (try resolving conflicts or commit local changes)
)
pause
goto MAIN_MENU

:NEW_BRANCH
echo.
set /p "BRANCH_NAME=Enter new branch name: "
if not "%BRANCH_NAME%"=="" (
    git checkout -b "%BRANCH_NAME%"
    if errorlevel 1 (
        echo Branch creation failed
    ) else (
        echo Switched to new branch: %BRANCH_NAME%
    )
)
pause
goto MAIN_MENU

:SWITCH_BRANCH
echo.
echo Available branches:
git branch
echo.
set /p "TARGET_BRANCH=Enter target branch: "
if not "%TARGET_BRANCH%"=="" (
    git checkout "%TARGET_BRANCH%"
    if errorlevel 1 (
        echo Branch switch failed
    ) else (
        echo Switched to branch: %TARGET_BRANCH%
    )
)
pause
goto MAIN_MENU

:MERGE_BRANCH
echo.
echo Available branches:
git branch
echo.
set /p "MERGE_BRANCH=Enter branch to merge: "
if not "%MERGE_BRANCH%"=="" (
    echo Merging branch: %MERGE_BRANCH%
    git merge "%MERGE_BRANCH%"
    if errorlevel 1 (
        echo Merge conflicts detected
    ) else (
        echo Merge completed
    )
)
pause
goto MAIN_MENU

:LOG_HISTORY
echo.
echo Recent 10 commits:
git log --oneline --graph --all --decorate -10
pause
goto MAIN_MENU

:RESET_SOFT
echo.
set /p "CONFIRM=Soft reset (keep changes)? y/N: "
if /i "%CONFIRM%"=="y" (
    git reset --soft HEAD~1
    if errorlevel 1 (
        echo Soft reset failed
    ) else (
        echo Soft reset completed
        echo Current status:
        git status -sb
    )
) else (
    echo Operation cancelled
)
pause
goto MAIN_MENU

:RESET_HARD
echo.
echo WARNING: DANGEROUS OPERATION
echo Hard reset will:
echo - Permanently discard all uncommitted changes
echo - Cannot undo file modifications
echo - Reset to previous commit state
echo.
set /p "CONFIRM=Confirm hard reset? y/N: "
if /i "%CONFIRM%"=="y" (
    set /p "FINAL_CONFIRM=FINAL CONFIRM: Discard all changes? y/N: "
    if /i "%FINAL_CONFIRM%"=="y" (
        git reset --hard HEAD~1
        if errorlevel 1 (
            echo Hard reset failed
        ) else (
            echo Hard reset completed
            echo Current status:
            git status -sb
        )
    ) else (
        echo Operation cancelled
    )
) else (
    echo Operation cancelled
)
pause
goto MAIN_MENU

:CREDENTIAL_MANAGER
echo.
git config --global credential.helper manager-core
if errorlevel 1 (
    echo Credential manager setup failed
) else (
    echo Credential manager enabled
    echo Login window will appear on next push
)
pause
goto MAIN_MENU

:SHOW_BRANCHES
echo.
echo All branches:
git branch -a
pause
goto MAIN_MENU

:DELETE_BRANCH
echo.
echo Available branches:
git branch
echo.
set /p "DELETE_BRANCH=Enter branch to delete: "
if not "%DELETE_BRANCH%"=="" (
    set /p "CONFIRM_DELETE=Delete branch %DELETE_BRANCH%? y/N: "
    if /i "%CONFIRM_DELETE%"=="y" (
        git branch -d "%DELETE_BRANCH%" 2>nul || (
            set /p "FORCE_DELETE=Branch not fully merged, force delete? y/N: "
            if /i "%FORCE_DELETE%"=="y" (
                git branch -D "%DELETE_BRANCH%"
                if errorlevel 1 (
                    echo Branch delete failed
                ) else (
                    echo Branch force deleted
                )
            ) else (
                echo Delete cancelled
            )
        )
        if not errorlevel 1 (
            echo Branch deleted
        )
    ) else (
        echo Delete cancelled
    )
)
pause
goto MAIN_MENU

:: ========== Helpers ==========

:: Overwrite tracking safely: unset-all then set once
:: Usage: call :set_tracking branchName
:set_tracking
set "BR=%~1"
if "%BR%"=="" exit /b 1
git config --unset-all branch.%BR%.remote >nul 2>nul
git config --unset-all branch.%BR%.merge  >nul 2>nul
git config branch.%BR%.remote origin
git config branch.%BR%.merge  refs/heads/%BR%
echo Tracking set: branch.%BR%.remote=origin / merge=refs/heads/%BR%
exit /b 0

:: Ensure origin exists (ask URL only if missing)
:ensure_origin
git remote get-url origin >nul 2>nul && exit /b 0
echo.
echo 尚未設定遠端 origin。
set "REMOTE_URL="
set /p "REMOTE_URL=請貼上遠端 URL (例如 https://github.com/you/repo.git): "
if "%REMOTE_URL%"=="" exit /b 1
git remote add origin "%REMOTE_URL%" || exit /b 1
echo 已設定 origin: %REMOTE_URL%
exit /b 0

:EXIT
echo.
echo Thank you for using Git Menu Tool!
echo Last working directory: %cd%
timeout /t 2 >nul
exit /b 0




