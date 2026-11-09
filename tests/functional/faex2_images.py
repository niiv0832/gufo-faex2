"""Обезличенные PNG-фикстуры FAEX2; только стандартная библиотека Python.

JPEG/WebP и ошибочные data URL берутся из image_inputs.image_cases() и
image_inputs.invalid_image_cases(), которым передаётся image_content.
Импорт полного openai_sdk и Pillow для этого не нужен.
"""

import base64
import struct
import zlib


def _png_content(width, height, pixels):
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    row_bytes = width * 3
    raw = b"".join(b"\0" + pixels[start:start + row_bytes]
                   for start in range(0, height * row_bytes, row_bytes))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, level=9))
           + chunk(b"IEND", b""))
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}


def image_content(color):
    """Вернуть одноцветный PNG 128×128 как блок OpenAI Chat image_url."""
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    if color not in colors:
        raise ValueError("Поддерживаются только цвета red и blue")
    return _png_content(128, 128, bytes(colors[color]) * 128 * 128)


def text_image_content():
    """Вернуть точную строку GUFO 742, чёрную на белом PNG 640×192.

    Собственный фиксированный шрифт 5×7, масштаб 12, без системных шрифтов.
    Эталон SHA-256 RGB-пикселей (без PNG-контейнера и фильтров строк):
    55056333621b13a0ea93250431999f4bc8e19714b8fabe42286de2623a816f50.
    """
    glyphs = {
        "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
        "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
        "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        " ": ("00000",) * 7,
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    }
    width, height, scale = 640, 192, 12
    text = "GUFO 742"
    left = (width - (len(text) * 6 - 1) * scale) // 2
    top = (height - 7 * scale) // 2
    pixels = bytearray(b"\xff" * (width * height * 3))
    for index, character in enumerate(text):
        for row, pattern in enumerate(glyphs[character]):
            for column, bit in enumerate(pattern):
                if bit != "1":
                    continue
                x = left + (index * 6 + column) * scale
                for y in range(top + row * scale, top + (row + 1) * scale):
                    start = (y * width + x) * 3
                    pixels[start:start + scale * 3] = b"\0" * (scale * 3)
    return _png_content(width, height, pixels)
