# -*- coding: utf-8 -*-
# 產生：venv_cheatsheet_full.pdf（繁體中文，無亂碼）
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# 使用我們做好的字型註冊模組（請確保 fonts/ 內有 NotoSansTC / NotoSansSymbols）
from pdf_fonts import ensure_fonts_registered

# 1) 一次註冊字型（NotoSansTC / NotoSansSymbols / 可選 SegoeUIEmoji）
resolved = ensure_fonts_registered()
BASE_FONT = "NotoSansTC"  # 全檔統一用這個，最穩

# 2) 樣式（全部使用 Unicode TTF，避免亂碼/噴錯）
H1 = ParagraphStyle("H1", fontName=BASE_FONT, fontSize=18,
                    leading=22, spaceAfter=10)
H2 = ParagraphStyle("H2", fontName=BASE_FONT,
                    fontSize=14, leading=18, spaceAfter=8)
BODY = ParagraphStyle("BODY", fontName=BASE_FONT, fontSize=11.5, leading=16)
MONO = ParagraphStyle("MONO", fontName=BASE_FONT,
                      fontSize=10.5, leading=14)  # 不用 Courier 以免編碼問題


def codeblock(lines):
    # 用 Unicode 安全的字型畫等寬區塊（不追求完美等寬，先以穩為主）
    return Preformatted("\n".join(lines).strip(), MONO)


# 3) 內容
sections = [
    ("一、建立／啟用／停用（PowerShell）", [
        "py -3.11 -m venv .venv",
        r".\.venv\Scripts\Activate.ps1",
        "deactivate",
    ]),
    ("（PowerShell 第一次若被擋，執行）", [
        "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned",
    ]),
    ("一、建立／啟用／停用（CMD）", [
        "py -3.11 -m venv .venv",
        r".\.venv\Scripts\activate",
        "deactivate",
    ]),
    ("一、在 VS Code", [
        "打開專案資料夾 → Ctrl+` 開終端機",
        "若 .vscode/settings.json 已指定 .venv，通常會自動啟用；否則手動執行上方指令",
    ]),
    ("二、常用檢查", [
        "python --version",
        "where python",
        # 避免 emoji 導致編碼問題；若想要符號可改成 ✓
        r'python -c "import PyQt6,pyqtgraph,serial,numpy,pandas; print(\'OK\')"',
    ]),
    ("三、安裝／同步依賴", [
        "python -m pip install --upgrade pip",
        "pip install -r requirements.txt",
        "pip freeze > requirements.txt",
    ]),
    ("四、日常 SOP（換機器）", [
        "py -3.11 -m venv .venv",
        r".\.venv\Scripts\Activate.ps1",
        "pip install -r requirements.txt",
    ]),
    ("五、常見問題速解", [
        "沒出現 (.venv) → 還沒啟用",
        "PowerShell 不能執行 Activate.ps1 → Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned",
        "用到系統 Python 而非 venv → where python 應看到 .venv",
        "安裝出現 MSVC / Build Tools 錯誤 → 先升級 pip；仍不行再裝 VS2022 Build Tools",
        "py 指令不存在 → 改用 python -m venv .venv",
        "VS Code 抓錯 → Ctrl+Shift+P → Python: Select Interpreter → 選 .venv",
    ]),
]

# 4) 輸出 PDF
pdf_out = "venv_cheatsheet_full.pdf"
doc = SimpleDocTemplate(
    pdf_out, pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=1.5*cm, bottomMargin=1.5*cm
)

story = [Paragraph("Windows 桌機／筆電通用 venv 啟動小抄", H1), Spacer(1, 0.2*cm)]
for title, lines in sections:
    story.append(Paragraph(title, H2))
    story.append(codeblock(lines))
    story.append(Spacer(1, 0.3*cm))

doc.build(story)
print("✅ 字型：", resolved)
print(f"✅ 已產生 PDF：{os.path.abspath(pdf_out)}")
