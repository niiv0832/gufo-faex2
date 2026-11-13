"""Сложные синтетические изображения приёмки доработок; только стандартная библиотека.

Абзац текста, формула и таблица чисел строятся независимо от модели собственным
растровым шрифтом 5×7 (масштаб 4: высота буквы 28 пикселей — ниже известной границы
отказа крупных букв, см. faex2_vision_probes.py и docs/faex2/INVESTIGATION_RESULT.md).
Эталон ответа известен до запроса. Для этих изображений нет сверки с эталонным
движком, поэтому неверный ответ остаётся FAIL и требует сравнения с контрольным
бинарником на тех же байтах; известным ограничением он не объявляется.
"""

import hashlib
import re

from faex2_vision_probes import BLACK, BLUE, GLYPHS, GREEN, RED, WHITE, _Canvas

FONT = {
    **GLYPHS,
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "=": ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "/": ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
}
SCALE = 4
ADVANCE = 6 * SCALE
LINE = 10 * SCALE

PARAGRAPH = ("RIVER PORT 583", "NINE OWLS WATCH", "THE NORTH GATE")
FORMULA = "X = (48 + 6) / 9"
TABLE = ((417, 82, 936), (25, 604, 71), (358, 19, 740))


def _draw(canvas, left, top, text):
    for index, character in enumerate(text):
        for row, pattern in enumerate(FONT[character]):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    x, y = left + index * ADVANCE + column * SCALE, top + row * SCALE
                    canvas.rect(x, y, x + SCALE, y + SCALE, BLACK)
    return canvas


def paragraph_image():
    canvas = _Canvas(448, 160)
    for index, line in enumerate(PARAGRAPH):
        _draw(canvas, 32, 24 + index * LINE, line)
    return canvas.content()


def formula_image():
    return _draw(_Canvas(448, 96), 32, 34, FORMULA).content()


def table_image():
    """Три строки и три столбца; линии сетки толщиной 2 пикселя, числа по левому краю ячеек."""
    cell_w, cell_h, left, top = 128, 48, 32, 24
    canvas = _Canvas(448, 192)
    for index in range(4):
        y, x = top + index * cell_h, left + index * cell_w
        canvas.rect(left, y, left + 3 * cell_w + 2, y + 2, BLACK)
        canvas.rect(x, top, x + 2, top + 3 * cell_h + 2, BLACK)
    for row, values in enumerate(TABLE):
        for column, value in enumerate(values):
            _draw(canvas, left + column * cell_w + 16, top + row * cell_h + 10, str(value))
    return canvas.content()


def colour_image(colour):
    return _Canvas(128, 128, {"red": RED, "blue": BLUE, "green": GREEN, "white": WHITE}[colour]).content()


def _compact(value):
    return re.sub(r"\s+", "", value.strip().rstrip(".")).upper()


def _lines(value):
    return [re.sub(r"\s+", " ", line.strip().rstrip(".")).upper()
            for line in value.splitlines() if line.strip()]


def complex_probes():
    """id, content (блок image_url), question, grade(answer) -> bool, expected (для отчёта)."""
    second_row = ",".join(str(value) for value in TABLE[1])
    return [
        {"id": "paragraph-lines", "content": paragraph_image(), "expected": "\n".join(PARAGRAPH),
         "question": "Read every line of text in this image. Reply with the lines only, "
                     "one per line, in the same order.",
         "grade": lambda answer: _lines(answer) == list(PARAGRAPH)},
        {"id": "formula-transcribe", "content": formula_image(), "expected": FORMULA,
         "question": "Transcribe the formula in this image exactly. Reply with the formula only.",
         "grade": lambda answer: _compact(answer) == _compact(FORMULA)},
        {"id": "formula-value", "content": formula_image(), "expected": "6",
         "question": "The image shows a formula for X. Compute X. Reply with the number only.",
         "grade": lambda answer: _compact(answer) == "6"},
        {"id": "table-second-row", "content": table_image(), "expected": second_row,
         "question": "The image is a table of numbers with three rows and three columns. "
                     "Reply with the three numbers of the SECOND row, comma-separated, nothing else.",
         "grade": lambda answer: _compact(answer) == second_row},
        {"id": "table-cell", "content": table_image(), "expected": str(TABLE[2][0]),
         "question": "The image is a table of numbers with three rows and three columns. "
                     "Which number is in the THIRD row, FIRST column? Reply with the number only.",
         "grade": lambda answer: _compact(answer) == str(TABLE[2][0])},
    ]


def image_sha(content):
    return hashlib.sha256(content["image_url"]["url"].encode()).hexdigest()
