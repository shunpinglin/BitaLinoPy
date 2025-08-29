# pdf_fonts.py  —— 修正版
from __future__ import annotations
import os
import glob
from typing import Dict, Tuple, List

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import Paragraph  # 只保留需要的 platypus 類別
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # ← 正確匯入

from reportlab.pdfgen import canvas

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

FONT_PATTERNS = {
    "NotoSansTC-Regular": ["NotoSansTC-Regular.ttf", "NotoSansTC*.ttf", "Noto Sans TC*.ttf"],
    "NotoSansTC-Bold":    ["NotoSansTC-Bold.ttf", "NotoSansTC*Bold*.ttf"],
    "NotoSansSymbols":    ["NotoSansSymbols-Regular.ttf", "NotoSansSymbols*.ttf", "Noto Sans Symbols*.ttf"],
    "SegoeUIEmoji":       ["SegoeUIEmoji.ttf", "seguiemj.ttf", "Segoe UI Emoji*.ttf"],
}


def _find_font_file(patterns: List[str]) -> str | None:
    for p in patterns:
        for path in glob.glob(os.path.join(FONTS_DIR, p)):
            if os.path.isfile(path):
                return path
    return None


def _register_ttf(name: str, path: str) -> None:
    if not path:
        return
    if name in pdfmetrics.getRegisteredFontNames():
        return
    pdfmetrics.registerFont(TTFont(name, path))


def ensure_fonts_registered() -> Dict[str, str]:
    os.makedirs(FONTS_DIR, exist_ok=True)
    resolved = {}

    path_tc_reg = _find_font_file(FONT_PATTERNS["NotoSansTC-Regular"])
    path_tc_bold = _find_font_file(FONT_PATTERNS["NotoSansTC-Bold"])
    path_symbols = _find_font_file(FONT_PATTERNS["NotoSansSymbols"])
    path_emoji = _find_font_file(FONT_PATTERNS["SegoeUIEmoji"])

    if path_tc_reg:
        _register_ttf("NotoSansTC", path_tc_reg)
        resolved["NotoSansTC"] = path_tc_reg
    else:
        raise FileNotFoundError("找不到 Noto Sans TC Regular（請放到 fonts/ 內）")

    if path_tc_bold:
        _register_ttf("NotoSansTC-Bold", path_tc_bold)
        resolved["NotoSansTC-Bold"] = path_tc_bold
    else:
        resolved["NotoSansTC-Bold"] = "(未提供，將以 Regular 代用)"

    if path_symbols:
        _register_ttf("NotoSansSymbols", path_symbols)
        resolved["NotoSansSymbols"] = path_symbols
    else:
        raise FileNotFoundError("找不到 Noto Sans Symbols（請放到 fonts/ 內）")

    if path_emoji:
        _register_ttf("SegoeUIEmoji", path_emoji)
        resolved["SegoeUIEmoji"] = path_emoji
    else:
        resolved["SegoeUIEmoji"] = "(未提供)"

    pdfmetrics.registerFontFamily(
        "NotoSansTC",
        normal="NotoSansTC",
        bold="NotoSansTC-Bold" if path_tc_bold else "NotoSansTC",
        italic="NotoSansTC",
        boldItalic="NotoSansTC-Bold" if path_tc_bold else "NotoSansTC",
    )
    addMapping("NotoSansTC", 0, 0, "NotoSansTC")
    addMapping("NotoSansTC", 1, 0,
               "NotoSansTC-Bold" if path_tc_bold else "NotoSansTC")

    return resolved


def get_zh_styles(base: str = "NotoSansTC", symbols: str = "NotoSansSymbols") -> Dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    zh = ParagraphStyle(
        "ZH", parent=styles["Normal"], fontName=base, fontSize=12, leading=16)
    zh_b = ParagraphStyle("ZH-B", parent=zh, fontName=base)
    sym = ParagraphStyle(
        "SYM", parent=styles["Normal"], fontName=symbols, fontSize=12, leading=16)
    code = ParagraphStyle("CODE", parent=styles.get(
        "Code", styles["Normal"]), fontName=base, fontSize=10, leading=14)
    return {"ZH": zh, "ZH-B": zh_b, "SYM": sym, "CODE": code}

# 簡易 fallback（Canvas）


def _pick_font_for_char(ch: str) -> str:
    if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f':
        return "NotoSansTC"
    o = ord(ch)
    if (0x2190 <= o <= 0x21FF) or (0x25A0 <= o <= 0x25FF) or (0x2600 <= o <= 0x27BF) or o >= 0x1F300:
        return "NotoSansSymbols"
    return "NotoSansTC"


def draw_string_with_fallback(c: canvas.Canvas, x: float, y: float, text: str, size: int = 12):
    if not text:
        return
    runs: List[Tuple[str, str]] = []
    cur_font = _pick_font_for_char(text[0])
    buf = [text[0]]
    for ch in text[1:]:
        f = _pick_font_for_char(ch)
        if f == cur_font:
            buf.append(ch)
        else:
            runs.append((cur_font, ''.join(buf)))
            cur_font, buf = f, [ch]
    runs.append((cur_font, ''.join(buf)))
    cx = x
    for f, s in runs:
        c.setFont(f, size)
        c.drawString(cx, y, s)
        cx += c.stringWidth(s, f, size)
