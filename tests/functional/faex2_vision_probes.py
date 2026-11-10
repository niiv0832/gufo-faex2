"""Независимые зрительные пробы FAEX2; только стандартная библиотека Python.

Набор отделён от исходной фикстуры `GUFO 742` (faex2_images.text_image_content),
которая сохранена побайтно как первая проба. Он различает три вопроса:

* читается ли один и тот же пиксельный текст при разном размере букв;
* сохранены ли порядок и геометрия патчей (половины, квадранты, полосы,
  положение метки, счёт), в том числе для неквадратных и некратных 32 размеров;
* совпадает ли поведение движка с эталонным на одних и тех же файлах.

`known_limit=True` помечает пробы, на которых Qwen3.6-35B-A3B UD-Q4_K_XL с
mmproj BF16 отказывает одинаково в Gufo и в llama.cpp (крупные буквы высотой
от ~56 пикселей). Такой отказ остаётся отказом: метка только запрещает
приписывать его пути изображения Gufo без сверки с эталоном. Результаты и
границы — docs/faex2/INVESTIGATION_RESULT.md.
"""

import re

from faex2_images import _png_content, text_image_content

RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 220, 0)
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
READ = "Read the text in this image. Reply with the text only, preserving the space."
CORNER = ("In which corner of the image is the black square? Answer with exactly one of: "
          "top-left, top-right, bottom-left, bottom-right.")
GLYPHS = {
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    " ": ("00000",) * 7,
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
}


class _Canvas:
    def __init__(self, width, height, colour=WHITE):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(colour) * (width * height))

    def rect(self, left, top, right, bottom, colour):
        for y in range(top, bottom):
            start = (y * self.width + left) * 3
            self.pixels[start:start + (right - left) * 3] = bytes(colour) * (right - left)
        return self

    def content(self):
        return _png_content(self.width, self.height, bytes(self.pixels))


def _pixel_text(width, height, scale, ink=BLACK, paper=WHITE, text="GUFO 742"):
    canvas = _Canvas(width, height, paper)
    left = (width - (len(text) * 6 - 1) * scale) // 2
    top = (height - 7 * scale) // 2
    for index, character in enumerate(text):
        for row, pattern in enumerate(GLYPHS[character]):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    x, y = left + (index * 6 + column) * scale, top + row * scale
                    canvas.rect(x, y, x + scale, y + scale, ink)
    return canvas.content()


def _halves(width, height, first, second, vertical=False):
    canvas = _Canvas(width, height)
    if vertical:
        return canvas.rect(0, 0, width, height // 2, first).rect(0, height // 2, width, height, second).content()
    return canvas.rect(0, 0, width // 2, height, first).rect(width // 2, 0, width, height, second).content()


def _quadrants():
    return (_Canvas(448, 448).rect(0, 0, 224, 224, RED).rect(224, 0, 448, 224, GREEN)
            .rect(0, 224, 224, 448, BLUE).rect(224, 224, 448, 448, YELLOW).content())


def _squares(width, height, count, vertical=False):
    canvas = _Canvas(width, height)
    for index in range(count):
        if vertical:
            canvas.rect(64, 96 + index * 192, 128, 160 + index * 192, BLACK)
        else:
            canvas.rect(60 + index * 112, 64, 124 + index * 112, 128, BLACK)
    return canvas.content()


def _stripes(colours, vertical_stack=False):
    canvas = _Canvas(160, 640) if vertical_stack else _Canvas(640, 160)
    for index, colour in enumerate(colours):
        if vertical_stack:
            canvas.rect(0, index * 160, 160, (index + 1) * 160, colour)
        else:
            canvas.rect(index * 160, 0, (index + 1) * 160, 160, colour)
    return canvas.content()


def probes():
    """Вернуть пробы: id, group, content (блок image_url), question, expected, known_limit."""
    def text(identifier, content, limit):
        return {"id": identifier, "group": "pixel-text", "content": content, "question": READ,
                "expected": "GUFO 742", "known_limit": limit}

    def spatial(identifier, content, question, expected):
        return {"id": identifier, "group": "spatial", "content": content, "question": question,
                "expected": expected, "known_limit": False}

    one_word = " Answer with one word."
    quadrants = _quadrants()
    return [
        # Исходная неуспешная фикстура — без изменений.
        text("pixel-s12-640x192", text_image_content(), True),
        text("pixel-s4-256x64", _pixel_text(256, 64, 4), False),
        text("pixel-s6-320x96", _pixel_text(320, 96, 6), False),
        text("pixel-s8-448x448", _pixel_text(448, 448, 8), True),
        text("pixel-s12-640x192-inverted", _pixel_text(640, 192, 12, WHITE, BLACK), True),
        text("pixel-s16-896x256", _pixel_text(896, 256, 16), True),
        spatial("left-red-right-blue-640x192", _halves(640, 192, RED, BLUE),
                "What colour is the LEFT half of this image?" + one_word, "red"),
        spatial("left-blue-right-red-640x192", _halves(640, 192, BLUE, RED),
                "What colour is the LEFT half of this image?" + one_word, "blue"),
        spatial("top-green-bottom-yellow-192x640", _halves(192, 640, GREEN, YELLOW, vertical=True),
                "What colour is the TOP half of this image?" + one_word, "green"),
        spatial("left-red-right-blue-500x300", _halves(500, 300, RED, BLUE),
                "What colour is the RIGHT half of this image?" + one_word, "blue"),
        spatial("quadrants-448x448-bottom-left", quadrants,
                "What colour is the BOTTOM-LEFT quadrant of this image?" + one_word, "blue"),
        spatial("quadrants-448x448-top-right", quadrants,
                "What colour is the TOP-RIGHT quadrant of this image?" + one_word, "green"),
        spatial("square-top-right-576x576", _Canvas(576, 576).rect(432, 48, 528, 144, BLACK).content(),
                CORNER, "top-right"),
        spatial("square-bottom-left-576x576", _Canvas(576, 576).rect(48, 432, 144, 528, BLACK).content(),
                CORNER, "bottom-left"),
        spatial("five-squares-640x192", _squares(640, 192, 5),
                "How many black squares are in this image? Answer with a number only.", "5"),
        spatial("three-squares-192x640", _squares(192, 640, 3, vertical=True),
                "How many black squares are in this image? Answer with a number only.", "3"),
        spatial("stripes-left-to-right-640x160", _stripes((RED, GREEN, BLUE, YELLOW)),
                "List the colours of the vertical stripes from left to right, comma-separated, lowercase.",
                "red, green, blue, yellow"),
        spatial("stripes-top-to-bottom-160x640", _stripes((YELLOW, BLUE, GREEN, RED), vertical_stack=True),
                "List the colours of the horizontal stripes from top to bottom, comma-separated, lowercase.",
                "yellow, blue, green, red"),
    ]


def grade(answer, expected):
    """Точное совпадение после обрезки пробелов, регистра и завершающей точки."""
    def normal(value):
        return re.sub(r"\s+", " ", value.strip().rstrip(".").lower())
    return normal(answer) == normal(expected)


def request(case, model="qwen3.6-35b-a3b"):
    """Greedy-запрос без рассуждений и без повторного использования кэша."""
    return {"method": "POST", "path": "/v1/chat/completions", "body": {
        "model": model, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [case["content"], {"type": "text", "text": case["question"]}]}],
        "max_completion_tokens": 48, "stream": False, "temperature": 0, "seed": 0}}
