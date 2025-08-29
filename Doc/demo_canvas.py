# demo_canvas.py
from reportlab.pdfgen import canvas
from pdf_fonts import ensure_fonts_registered, draw_string_with_fallback

ensure_fonts_registered()

c = canvas.Canvas("demo_canvas.pdf")
draw_string_with_fallback(c, 72, 770, "中文OK：心率變異性（HRV）✓ ✨ ▶︎ 😀", size=14)
c.save()
