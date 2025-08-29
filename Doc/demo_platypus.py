# demo_platypus.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pdf_fonts import ensure_fonts_registered, get_zh_styles

ensure_fonts_registered()
styles = get_zh_styles()

doc = SimpleDocTemplate("demo_platypus.pdf")
story = [
    Paragraph("中文段落：這是一段以 Noto Sans TC 顯示的中文。", styles["ZH"]),
    Spacer(1, 8),
    Paragraph("<b>粗體測試：</b>這裡會自動切到 NotoSansTC-Bold（若提供）。", styles["ZH"]),
    Spacer(1, 8),
    Paragraph("符號示例：✓ ✔ ✨ ★ ▶︎ ⌘ ⌛", styles["SYM"]),
    Spacer(1, 8),
    Paragraph(
        '<font face="NotoSansTC">混排：</font><font face="NotoSansSymbols">★ ✨ ▶︎</font>', styles["ZH"]),
]
doc.build(story)
