一、一次性設定（建議先做）
# 基本身分（只需一次）
git config --global user.name  "你的名字"
git config --global user.email "你的GitHub email"

# 讓中文檔名正常顯示
git config --global core.quotepath false
git config --global i18n.commitEncoding utf-8
git config --global i18n.logOutputEncoding utf-8

# 習慣用 rebase 同步（避免多餘 merge commit）
git config --global pull.rebase true

二、把現有資料夾推上 GitHub
# 進入專案資料夾（Windows 路徑要寫 /c/...）
cd /c/Users/BigLin/Desktop/BitaLinoPy

git init
git branch -M main

# 第一次提交
git add .
git commit -m "chore: initial commit"

# 綁定遠端（改成你的帳號）
git remote add origin https://github.com/<你的帳號>/BitaLinoPy.git
git push -u origin main

若被拒（因遠端已有 README/.gitignore）

目標：保留本地 .gitignore，其他照情況合併

git fetch origin
git pull --rebase origin main     # 進入 rebase 流程

# 若只有 .gitignore 衝突：保留本地版本
git checkout --ours .gitignore
git add .gitignore
git rebase --continue

# 有其他檔案衝突 → 逐一選 ours(本地) / theirs(遠端)
#   例：保留本地 README
#   git checkout --ours README.md && git add README.md && git rebase --continue

# 完成後推上去
git push -u origin main

三、日常開發流程（最常用）
# 查看狀態
git status

# 加入所有變更並提交
git add .
git commit -m "feat: 說明你做了什麼"

# 先拉再推（避免衝突）
git pull --rebase origin main
git push

四、分支（開新功能、開 PR）
# 開新分支
git checkout -b feature/pdf-fonts

# 開發 → 提交 → 推到遠端
git add .
git commit -m "feat: 新增 pdf_fonts 註冊模組"
git push -u origin feature/pdf-fonts

# 完成後到 GitHub 開 Pull Request

五、臨時收納未提交變更（要切分支/換工作）
git stash              # 收起來
git switch 其它分支
# 回來時
git switch 原分支
git stash pop          # 取出並套回

六、常見急救指令
# 取消工作目錄未加入暫存區的變更
git restore .

# 取消已加入暫存區的變更
git reset

# 查看最近 10 筆提交（簡潔圖形）
git log --oneline --graph -n 10

# rebase 卡住想重來
git rebase --abort

# 遇到 push 被拒（non-fast-forward）：先 rebase，再推
git pull --rebase origin main
git push

# 真的只想用本地覆蓋遠端（單人專案才用）
git push --force-with-lease

七、大檔案（>100MB）用 Git LFS（選用）
# 安裝並追蹤常見二進位
git lfs install
git lfs track "*.zip" "*.mp4" "*.wav" "*.pptx" "*.ttf" "*.otf"
git add .gitattributes
git commit -m "chore: track large binaries via LFS"
git push

八、.gitignore 建議重點（Python 專案）
__pycache__/
*.py[cod]
.dist-info/
.ipynb_checkpoints/
.venv/
venv/
.env
.env.*
.vscode/
!.vscode/settings.json
!.vscode/extensions.json
!.vscode/tasks.json
*.log
*.tmp
# 需要的話再忽略：# *.pdf
fonts/private/
fonts/SegoeUIEmoji*.ttf
