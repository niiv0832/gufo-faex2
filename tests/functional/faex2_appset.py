"""Генераторы и реестр прикладного набора качества FAEX2 (Б2); только stdlib.

Домены A–D из docs/superpowers/specs/2026-11-19-b2-quality-set-q1-design.md §3:
84 новые задачи; 12 существующих домена A читаются раннером из
faex2_quality.jsonl. Все тексты и изображения синтетические и
воспроизводимые: никаких личных данных, никакой сети, никакого Pillow.

Задачи с изображениями используют только символы, которые есть в
закреплённом шрифте 5x7 (faex2_remediation_images.FONT): латиница,
цифры, пробел и "+ = ( ) /".
"""

from __future__ import annotations

import json
from pathlib import Path

from faex2_remediation_images import ADVANCE, FONT, LINE, SCALE, _draw
from faex2_images import _png_content
from faex2_vision_probes import BLACK, WHITE, _Canvas, _pixel_text

# Оценка «символов на токен» для калибровки длинных документов.
# Точная длина на стенде проверяется фактическим prompt_tokens (POST /tokenize
# или input_tokens); эта константа нужна только для детерминированной генерации.
CHARS_PER_TOKEN = 2.4
SPARSE_THRESHOLD = 18_560

_WINDS = ("С", "СВ", "В", "ЮВ", "Ю", "ЮЗ", "З", "СЗ")
_PRECIP = ("ясно", "облачно", "дождь", "снег", "туман")


# ---------------------------------------------------------------------------
# Длинные детерминированные документы (домен D)
# ---------------------------------------------------------------------------

def long_document(doc_index: int, target_tokens: int):
    """Детерминированный «метеодневник»: уникальный факт в каждой записи.

    Возвращает (текст, факты), где факты — {день: {"temp": int, "wind": str,
    "precip": str}}. Один и тот же вызов всегда даёт один и тот же текст.
    """
    _filler = "Показания сняты штатно, приборы проверены, отклонений нет. "
    _sample = "День 999: температура -99, ветер СЗ, осадки — туман. "
    fact_chars = len(_sample) + len(_filler)
    blocks = max(60, int(target_tokens * CHARS_PER_TOKEN / fact_chars))
    facts = {}
    parts = [f"МЕТЕОДНЕВНИК СТАНЦИИ K-{doc_index}. Записи за сентябрь.\n"]
    for day in range(1, blocks + 1):
        temp = ((day * 7 + doc_index * 3) % 19) - 8
        wind = _WINDS[(day + doc_index) % 8]
        precip = _PRECIP[(day * 5 + doc_index) % 5]
        facts[day] = {"temp": temp, "wind": wind, "precip": precip}
        parts.append(f"День {day}: температура {temp}, ветер {wind}, "
                     f"осадки — {precip}. ")
        parts.append(_filler)
    return "".join(parts), facts


def _fact_day(doc_index: int, position: str, target_tokens: int) -> int:
    """День для вопроса long-facts: начало/середина/конец документа."""
    _, facts = long_document(doc_index, target_tokens)
    days = sorted(facts)
    offset = {"begin": 0.1, "mid": 0.5, "end": 0.9}[position]
    return days[min(len(days) - 1, int(len(days) * offset))]


def fact_question(doc_index: int, position: str, target_tokens: int):
    """Вопрос long-facts с эталоном, известным до запроса."""
    day = _fact_day(doc_index, position, target_tokens)
    text, facts = long_document(doc_index, target_tokens)
    temp = facts[day]["temp"]
    second_day = day + 1
    second_temp = facts[second_day]["temp"]
    prompt = (f"Документ (метеодневник):\n\n{text}\n\n"
              f"Какая температура была в дни {day} и {second_day}? "
              "Ответь только массивом JSON из двух чисел в указанном порядке.")
    return prompt, {"day": day, "temp": temp,
                    "fact_checks": [[day, temp], [second_day, second_temp]]}


# ---------------------------------------------------------------------------
# Изображения (только символы шрифта 5x7: A-Z, 0-9, пробел, + = ( ) /)
# ---------------------------------------------------------------------------

def _text_canvas(lines, left=32, top=24, scale=SCALE):
    width = max(left * 2 + max(len(line) for line in lines) * ADVANCE + scale,
                320)
    height = top * 2 + len(lines) * LINE
    canvas = _Canvas(width, height)
    for index, line in enumerate(lines):
        _draw(canvas, left, top + index * LINE, line)
    return canvas.content()


def code_word_image(word: str) -> dict:
    return _text_canvas([word])


def big_text_image(word: str, scale: int) -> dict:
    """Крупный текст 48–96 pt (высота буквы 7×scale px): группа known_limit."""
    width = len(word) * 6 * scale + 2 * scale
    height = 9 * scale
    canvas = _Canvas(width, height)
    for index, character in enumerate(word):
        for row, pattern in enumerate(FONT[character]):
            for column, bit in enumerate(pattern):
                if bit == "1":
                    x = scale + (index * 6 + column) * scale
                    y = scale + row * scale
                    canvas.rect(x, y, x + scale, y + scale, BLACK)
    return canvas.content()


_TABLES = {
    "table-a": ((12, 34, 56), (78, 90, 11), (23, 45, 67)),
    "table-b": ((5, 10, 15), (20, 25, 30), (35, 40, 45)),
    "table-c": ((7, 14, 21), (28, 35, 42), (49, 56, 63)),
    "table-d": ((3, 6, 9, 12), (15, 18, 21, 24)),
    "table-e": ((100, 200), (300, 400), (500, 600)),
    "table-f": ((9, 8, 7), (6, 5, 4), (3, 2, 1)),
}


def table_image(name: str) -> dict:
    values = _TABLES[name]
    columns = max(len(row) for row in values)
    cell_w, cell_h, left, top = 128, 48, 24, 24
    canvas = _Canvas(left * 2 + columns * cell_w, top * 2 + len(values) * cell_h)
    for index in range(len(values) + 1):
        y = top + index * cell_h
        canvas.rect(left, y, left + columns * cell_w + 2, y + 2, BLACK)
    for index in range(columns + 1):
        x = left + index * cell_w
        canvas.rect(x, top, x + 2, top + len(values) * cell_h + 2, BLACK)
    for row, cells in enumerate(values):
        for column, value in enumerate(cells):
            _draw(canvas, left + column * cell_w + 16, top + row * cell_h + 10,
                  str(value))
    return canvas.content()


_RECEIPTS = {
    "receipt-a": (["SHOP 742", "BREAD 45", "MILK 92", "CHEESE 310"], 447),
    "receipt-b": (["SHOP 118", "TEA 130", "CAKE 260"], 390),
    "receipt-c": (["SHOP 66", "PEN 25", "BOOK 120", "BAG 210"], 355),
    "receipt-d": (["SHOP 9", "EGG 80"], 80),
}


def receipt_image(name: str) -> dict:
    lines, total = _RECEIPTS[name]
    return _text_canvas([*lines, f"TOTAL {total}"])


_FORMULAS = {"formula-a": ("X = (35 + 7) / 6", 7), "formula-b": ("X = (72 + 8) / 5", 16)}


def formula_image(name: str) -> dict:
    text, _ = _FORMULAS[name]
    return _text_canvas([text], top=34)


_IMAGES = {
    "code-river-583": ("RIVER 583", code_word_image),
    "code-owl-912": ("OWL 912", code_word_image),
    "code-gate-274": ("GATE 274", code_word_image),
    "big-gufo-518": None, "big-owl-274": None,
    **{f"table-{letter}": None for letter in "abcdef"},
    **{f"receipt-{letter}": None for letter in "abcd"},
    "formula-a": None, "formula-b": None,
}


def image(name: str) -> dict:
    """Блок OpenAI Chat image_url для фиксированного имени изображения."""
    if name in ("big-gufo-518", "big-owl-274"):
        return big_text_image("GUFO 518" if name.endswith("518") else "OWL 274",
                              12 if name.endswith("518") else 10)
    if name.startswith("table-"):
        return table_image(name)
    if name.startswith("receipt-"):
        return receipt_image(name)
    if name.startswith("formula-"):
        return formula_image(name)
    if name in ("code-river-583", "code-owl-912", "code-gate-274"):
        return code_word_image(_IMAGES[name][0])
    raise ValueError("Неизвестное изображение: " + name)


def image_names() -> list:
    return sorted(set(_TABLES) | set(_RECEIPTS) | set(_FORMULAS)
                  | {"code-river-583", "code-owl-912", "code-gate-274",
                     "big-gufo-518", "big-owl-274"})


# ---------------------------------------------------------------------------
# Реестр 84 задач
# ---------------------------------------------------------------------------

CRITICAL = ["incorrect_math", "incorrect_fact", "fabricated_source", "empty", "truncated"]

DEFAULT_RUBRIC = {
    "correctness": {"0": "Ответ неверный, пустой или содержит существенное противоречие.",
                    "1": "Часть необходимых фактов или шагов отсутствует.",
                    "2": "Все обязательные факты и значения верны."},
    "instruction": {"0": "Задача не выполнена.", "1": "Есть несущественное нарушение формы.",
                    "2": "Инструкция и заданная форма ответа выполнены."},
    "russian_clarity": {"0": "Ответ непонятен.", "1": "Ответ понятен, но содержит языковые ошибки.",
                        "2": "Ответ ясен; язык и формат соответствуют заданию."},
    "explanation": {"0": "Требуемое объяснение неверно или отсутствует.",
                    "1": "Требуемое объяснение неполное.",
                    "2": "Объяснение верно, если оно запрошено; иначе не требуется."},
}
DEFAULT_PASS = {"correctness": 2, "minimum_per_dimension": 1, "minimum_total": 6,
                "critical_failure_result": "FAIL"}


def _task(id_, domain, bucket, evaluation, thinking, **fields):
    if not fields.get("expected_facts"):
        if fields.get("expected_values"):
            fields["expected_facts"] = [
                "Зафиксированные значения: " +
                json.dumps(fields["expected_values"], ensure_ascii=False, sort_keys=True)]
        elif fields.get("autocheck"):
            fields["expected_facts"] = [
                "Фиксированные условия проверки: " +
                json.dumps(fields["autocheck"], ensure_ascii=False, sort_keys=True)]
        else:
            fields["expected_facts"] = [
                "Ответ должен опираться только на приведённый синтетический источник."]
    fields.setdefault("prohibited_errors", [
        "Выдуманные факты, противоречия исходным данным и невыполнение инструкции."])
    fields.setdefault("rubric", DEFAULT_RUBRIC)
    fields.setdefault("pass_criteria", DEFAULT_PASS)
    return {"id": id_, "domain": domain, "bucket": bucket, "evaluation": evaluation,
            "thinking": thinking, "critical_failures": list(CRITICAL), **fields}


def _ocr_text_tasks():
    paragraphs = {
        "ocr-text-1": ["BLUE LAKE 214", "COLD WIND RISES", "SEVEN HILLS"],
        "ocr-text-2": ["GOLD FIELD 905", "SILENT RIVER TURNS", "EAST GATE 41"],
        "ocr-text-3": ["PINE ROAD 66", "SOUTH LAKE FROST", "TWO OWLS SING"],
        "ocr-text-4": ["WEST MILL 730", "BIRDS LEAVE SOON", "GREEN HILL 9"],
        "ocr-text-5": ["NORTH STAR 118", "OLD BRIDGE WAITS", "RED TRAIN 5"],
        "ocr-text-6": ["SALT LAKE 802", "FINE DAY ENDS", "GATE 3 OPENS"],
        "ocr-text-7": ["HIGH WALL 47", "NIGHT BIRD FLIES", "COOL AIR 12"],
        "ocr-text-8": ["DEEP WELL 630", "LIGHT RAIN FALLS", "SOUTH WIND 8"],
    }
    tasks = []
    for id_, lines in paragraphs.items():
        image_name = "img-" + id_
        if id_ in ("ocr-text-1", "ocr-text-2", "ocr-text-5", "ocr-text-6"):
            tasks.append(_task(id_, "ocr-text", "short", "auto", "off",
                               images=[image_name],
                               prompt="Перепиши все три строки текста с изображения по порядку, "
                                      "каждая строка на своей строке, без пояснений.",
                               max_completion_tokens=96,
                               autocheck={"type": "line_match", "value": lines},
                               image_lines={image_name: lines},
                               expected_values={"lines": lines}))
        else:
            line_number, expected = {"ocr-text-3": (1, "66"), "ocr-text-4": (3, "9"),
                                     "ocr-text-7": (1, "47"), "ocr-text-8": (1, "630")}[id_]
            tasks.append(_task(id_, "ocr-text", "short", "auto", "off",
                               images=[image_name],
                               prompt=f"На изображении три строки текста. Какое число написано "
                                      f"в строке номер {line_number}? Ответь только числом.",
                               max_completion_tokens=48,
                               autocheck={"type": "compact", "value": expected},
                               image_lines={image_name: lines},
                               expected_values={"number": expected}))
    return tasks, {name: _text_canvas(lines) for name, lines in
                   ((("img-" + id_), lines) for id_, lines in paragraphs.items())}


def _number_line(values):
    return ", ".join(str(value) for value in values)


def _table_tasks():
    questions = {
        "table-a": ("Какое число в таблице находится в строке 2, столбце 3? "
                    "Ответь только числом.", "compact", "11"),
        "table-b": ("Сложи все девять чисел таблицы. Ответь только числом.", "numeric", 225),
        "table-c": ("Какое число в таблице находится в строке 3, столбце 1? "
                    "Ответь только числом.", "compact", "49"),
        "table-d": ("Сколько в таблице чисел больше 10? Ответь только числом.", "numeric", 5),
        "table-e": ("Сложи все числа таблицы. Ответь только числом.", "numeric", 2100),
        "table-f": ("Какое число в таблице находится в строке 1, столбце 2? "
                    "Ответь только числом.", "compact", "8"),
    }
    tasks = []
    for name, (question, kind, expected) in questions.items():
        tasks.append(_task("ocr-" + name, "ocr-table", "short", "auto", "off",
                           images=[name], prompt=question, max_completion_tokens=48,
                           autocheck={"type": kind, "value": expected},
                           expected_values={"answer": expected}))
    return tasks


def _receipt_tasks():
    questions = {
        "receipt-a": "На чеке три покупки и итоговая строка TOTAL. Какова итоговая сумма? Ответь только числом.",
        "receipt-b": "Какова итоговая сумма (TOTAL) в чеке? Ответь только числом.",
        "receipt-c": "Какова итоговая сумма (TOTAL) в чеке? Ответь только числом.",
        "receipt-d": "Какова итоговая сумма (TOTAL) в чеке? Ответь только числом.",
    }
    tasks = []
    for name, question in questions.items():
        total = _RECEIPTS[name][1]
        tasks.append(_task("ocr-" + name, "ocr-receipt", "short", "auto", "off",
                           images=[name], prompt=question, max_completion_tokens=48,
                           autocheck={"type": "numeric", "value": total},
                           expected_values={"total": total}))
    return tasks


def _formula_tasks():
    tasks = []
    for name, (_, value) in _FORMULAS.items():
        tasks.append(_task("ocr-" + name, "ocr-formula", "short", "auto", "off",
                           images=[name],
                           prompt="На изображении формула для X. Вычисли X. Ответь только числом.",
                           max_completion_tokens=48,
                           autocheck={"type": "numeric", "value": value},
                           expected_values={"x": value}))
    return tasks


def _multi_image_tasks():
    return [
        _task("vision-multi-1", "vision-multi", "short", "auto", "off",
              images=["code-river-583", "code-owl-912"],
              prompt="Даны две картинки с кодовыми словами. Какое кодовое слово написано "
                     "на ПЕРВОЙ картинке? Ответь только этим кодовым словом.",
              max_completion_tokens=48, autocheck={"type": "compact", "value": "RIVER 583"},
              expected_values={"answer": "RIVER 583"}),
        _task("vision-multi-2", "vision-multi", "short", "auto", "off",
              images=["code-owl-912", "code-gate-274"],
              prompt="Даны две картинки с кодовыми словами. Какое кодовое слово написано "
                     "на ПЕРВОЙ картинке? Ответь только этим кодовым словом.",
              max_completion_tokens=48, autocheck={"type": "compact", "value": "OWL 912"},
              expected_values={"answer": "OWL 912"}),
    ]


def _big_text_tasks():
    return [
        _task("big-text-gufo-518", "big-text", "short", "known_limit", "off",
              images=["big-gufo-518"],
              prompt="Напиши текст с картинки, сохраняя пробел.",
              max_completion_tokens=48, autocheck={"type": "compact", "value": "GUFO 518"},
              expected_values={"text": "GUFO 518"}, known_limit=True),
        _task("big-text-owl-274", "big-text", "short", "known_limit", "off",
              images=["big-owl-274"],
              prompt="Напиши текст с картинки, сохраняя пробел.",
              max_completion_tokens=48, autocheck={"type": "compact", "value": "OWL 274"},
              expected_values={"text": "OWL 274"}, known_limit=True),
    ]


_CODE_FIXES = [
    ("code-fix-sum-range", 15,
     "Код должен суммировать все целые числа от 1 до n включительно:\n"
     "    total = 0\n    for i in range(1, n):\n        total += i\n    print(total)\n"
     "В коде есть ровно одна ошибка. Мысленно исправь её и напиши, какое число выведет "
     "исправленный код при n = 5. Ответь только числом."),
    ("code-fix-sort-order", "[1, 2, 3]",
     "Функция должна возвращать список по возрастанию:\n"
     "    def f(a):\n        return sorted(a, reverse=True)\n"
     "В коде есть ровно одна ошибка. Мысленно исправь её и напиши, что вернёт исправленная "
     "функция для a = [3, 1, 2]. Ответь только списком."),
    ("code-fix-loop-bound", 2,
     "Код должен считать, сколько элементов списка [5, 12, 8, 20] больше 10:\n"
     "    count = 0\n    for i in range(len(a) - 1):\n        if a[i] > 10:\n            count += 1\n"
     "    print(count)\nВ коде есть ровно одна ошибка. Мысленно исправь её и напиши, какое "
     "число выведет исправленный код для a = [5, 12, 8, 20]. Ответь только числом."),
    ("code-fix-index", 7,
     "Функция должна возвращать ПЕРВЫЙ элемент списка:\n"
     "    def first(a):\n        return a[1]\n"
     "В коде есть ровно одна ошибка. Мысленно исправь её и напиши, что вернёт исправленная "
     "функция для a = [7, 4, 9]. Ответь только числом."),
    ("code-fix-product", 24,
     "Код должен вычислять произведение чисел списка:\n"
     "    p = 1\n    for i in [2, 3, 4]:\n        p = p + i\n    print(p)\n"
     "В коде есть ровно одна ошибка. Мысленно исправь её и напиши, какое число выведет "
     "исправленный код. Ответь только числом."),
    ("code-fix-slice", "АБВ",
     "Функция должна возвращать строку целиком:\n"
     "    def f(s):\n        return s[1:]\n"
     "В коде есть ровно одна ошибка. Мысленно исправь её и напиши, что вернёт исправленная "
     "функция для s = \"АБВ\". Ответь только строкой."),
]


def _code_tasks():
    tasks = []
    for id_, expected, prompt in _CODE_FIXES:
        kind = "numeric" if isinstance(expected, int) else "compact"
        tasks.append(_task(id_, "code-fix", "short", "auto", "off",
                           prompt=prompt, max_completion_tokens=64,
                           autocheck={"type": kind, "value": expected},
                           expected_values={"answer": expected}))
    tasks.append(_task("code-explain-last", "code-explain", "short", "auto", "off",
                       prompt="Что вернёт функция\n    def g(a):\n            return a[-2]\n"
                              "при a = [1, 5, 9, 3]? Ответь только значением.",
                       max_completion_tokens=48,
                       autocheck={"type": "numeric", "value": 9},
                       expected_values={"answer": 9}))
    tasks.append(_task("code-explain-set", "code-explain", "short", "auto", "off",
                       prompt="Что вернёт выражение len(set([1, 2, 2, 3]))? Ответь только числом.",
                       max_completion_tokens=48,
                       autocheck={"type": "numeric", "value": 3},
                       expected_values={"answer": 3}))
    tasks.append(_task("code-explain-words-1", "code-explain", "short", "manual", "off",
                       prompt="Объясни школьнику простыми словами, что делает функция\n"
                              "    def h(a):\n        total = 0\n        for x in a:\n"
                              "            total += x\n        return total / len(a)\n"
                              "и чему равен её результат для a = [2, 4, 6].",
                       max_completion_tokens=384,
                       expected_values={"answer": 4}))
    tasks.append(_task("code-explain-words-2", "code-explain", "short", "manual", "off",
                       prompt="Объясни школьнику простыми словами, что делает функция\n"
                              "    def k(a):\n        best = a[0]\n        for x in a:\n"
                              "            if x > best:\n                best = x\n"
                              "        return best\n"
                              "и чему равен её результат для a = [5, 12, 3].",
                       max_completion_tokens=384,
                       expected_values={"answer": 12}))
    return tasks


def _tool_chain_tasks():
    def schema(name, description, field, values):
        return [{"type": "function", "function": {
            "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": {field: {"type": "string",
                            "enum": list(values)}}, "required": [field],
                           "additionalProperties": False}}}]

    tasks = []
    for topic, answer in (("астрономия", "ПОЛЯРИС_742"), ("ботаника", "ФЛОРА_318")):
        tasks.append(_task(f"tool-chain-lookup-{topic}", "tool-chain", "medium", "auto",
                           "hermes", max_completion_tokens=768,
                           tools=schema("lookup", "Школьный справочник; внешних действий нет.",
                                        "topic", ("астрономия", "ботаника")),
                           first_prompt=f"Найди в справочнике тему «{topic}».",
                           tool_name="lookup", tool_arguments={"topic": topic},
                           tool_result={"answer": answer},
                           prompt="Назови только значение answer из результата инструмента.",
                           autocheck={"type": "compact", "value": answer},
                           expected_values={"answer": answer}))
    for city, temp in (("сочи", 20), ("казань", 15)):
        tasks.append(_task(f"tool-chain-weather-{city}", "tool-chain", "medium", "auto",
                           "hermes", max_completion_tokens=768,
                           tools=schema("get_weather", "Кэш школьного проекта о погоде; "
                                        "внешних запросов нет.", "city", ("сочи", "казань")),
                           first_prompt=f"Посмотри погоду для города «{city}».",
                           tool_name="get_weather", tool_arguments={"city": city},
                           tool_result={"temp_c": temp},
                           prompt="Переведи температуру из результата инструмента в градусы "
                                  "Фаренгейта. Ответь только числом.",
                           autocheck={"type": "numeric", "value": temp * 9 // 5 + 32},
                           expected_values={"fahrenheit": temp * 9 // 5 + 32}))
    for day, lesson in (("monday", "биология"), ("friday", "история")):
        tasks.append(_task(f"tool-chain-lesson-{day}", "tool-chain", "medium", "auto",
                           "hermes", max_completion_tokens=768,
                           tools=schema("get_lesson", "Расписание класса; внешних действий нет.",
                                        "day", ("monday", "friday")),
                           first_prompt=f"Посмотри в расписании, какой урок в {day}.",
                           tool_name="get_lesson", tool_arguments={"day": day},
                           tool_result={"lesson": lesson},
                           prompt="Какой урок в результате инструмента? Ответь одним словом.",
                           autocheck={"type": "compact", "value": lesson},
                           expected_values={"lesson": lesson}))
    for index, task in enumerate(tasks):
        preamble, _ = long_document(100 + index, 5000)
        task["first_prompt"] = ("Синтетический справочный архив:\n" + preamble +
                                "\nСледующее поручение относится к этому архиву:\n" +
                                task["first_prompt"])
        task["target_range"] = [2000, 18000]
        key = next(iter(task["tool_arguments"].values()))
        second = {"name": "confirm_record", "arguments": {"key": key},
                  "result": {"confirmed": True}}
        task["tools"].append({"type": "function", "function": {
            "name": "confirm_record", "description": "Проверяет синтетическую запись.",
            "strict": True, "parameters": {"type": "object", "properties": {
                "key": {"type": "string", "enum": [key]}},
                "required": ["key"], "additionalProperties": False}}})
        task["tool_sequence"] = [
            {"name": task["tool_name"], "arguments": task["tool_arguments"],
             "result": task["tool_result"]}, second]
        task["second_prompt"] = ("Проверь запись вторым инструментом confirm_record; "
                                 f"вызови его с key равным «{key}».")
    return tasks


_JSON_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "city_record", "strict": True,
    "schema": {"type": "object", "properties": {"city": {"type": "string"},
                                               "year": {"type": "integer"}},
               "required": ["city", "year"], "additionalProperties": False}}}


def _json_schema_tasks():
    return [
        _task("json-schema-tula", "json-schema", "short", "auto", "off",
              prompt="Верни JSON: city — слово «Тула», year — год 1146. Без пояснений.",
              max_completion_tokens=128, response_format=_JSON_SCHEMA,
              autocheck={"type": "json", "value": {"city": "Тула", "year": 1146}},
              expected_values={"city": "Тула", "year": 1146}),
        _task("json-schema-owl", "json-schema", "short", "auto", "off",
              prompt="Верни JSON: bird — слово «сова», count — число 9. Без пояснений.",
              max_completion_tokens=128, response_format={
                  "type": "json_schema", "json_schema": {
                      "name": "bird_count", "strict": True,
                      "schema": {"type": "object", "properties": {
                          "bird": {"type": "string"}, "count": {"type": "integer"}},
                          "required": ["bird", "count"], "additionalProperties": False}}},
              autocheck={"type": "json", "value": {"bird": "сова", "count": 9}},
              expected_values={"bird": "сова", "count": 9}),
    ]


def _hermes_tasks():
    return [
        _task("hermes-reuse-tool-01", "hermes-reuse", "medium", "auto", "hermes",
              hermes_case="tool-01-search", max_completion_tokens=None, target_range=[2000, 18000],
              autocheck={"type": "valid_tool_call", "name": "web_search",
                         "required_text": "query"},
              expected_values={"case": "tool-01-search"}),
        _task("hermes-reuse-mixed-02", "hermes-reuse", "medium", "auto", "hermes",
              hermes_case="mixed-02-ls-then-read", max_completion_tokens=None, target_range=[2000, 18000],
              autocheck={"type": "valid_tool_call", "name": "read_file",
                         "required_text": "path", "contains": "schedule.csv"},
              expected_values={"case": "mixed-02-ls-then-read"}),
    ]


def _a_new_tasks():
    tasks = [
        _task("form-three-sentences", "form-constraint", "short", "auto", "off",
              prompt="Объясни школьнику, что такое периметр многоугольника, ровно тремя "
                     "предложениями. Не больше и не меньше.",
              max_completion_tokens=384,
              autocheck={"type": "all", "checks": [
                  {"type": "sentence_count", "value": 3},
                  {"type": "contains_all", "values": ["периметр"]}]}),
        _task("form-four-bullets", "form-constraint", "short", "auto", "off",
              prompt="Перечисли четыре органа растения: корень, стебель, лист, цветок. Ответ — маркированный "
                     "список из ровно четырёх пунктов, без пояснений.",
              max_completion_tokens=192,
              autocheck={"type": "all", "checks": [
                  {"type": "bullet_count", "value": 4},
                  {"type": "contains_all", "values": ["корень", "стебель", "лист", "цветок"]}]}),
        _task("spelling-park", "spelling", "short", "auto", "off",
              prompt="Исправь орфографические и пунктуационные ошибки и запиши исправленный "
                     "текст целиком: «Мы пришло в парк. Трава была высокая? Солнце светило "
                     "ярче чем вчера.»",
              max_completion_tokens=256,
              autocheck={"type": "all", "checks": [
                  {"type": "contains_all", "values": ["пришли", "ярче, чем"]},
                  {"type": "excludes_all", "values": ["пришло", "?"]}]}),
        _task("spelling-forest", "spelling", "short", "auto", "off",
              prompt="Исправь орфографические ошибки и запиши исправленный текст целиком: "
                     "«Дети пошли в лис. В густом лесу росли грибы. Мы взяли корзину и ножници.»",
              max_completion_tokens=256,
              autocheck={"type": "all", "checks": [
                  {"type": "contains_all", "values": ["ножницы"]},
                  {"type": "excludes_all", "values": ["ножници", "в лис"]}]}),
        _task("dialog-corr-notebooks", "dialog-correction", "short", "auto", "off",
              messages=[{"role": "user", "content": "Купим 3 блокнота по 60 рублей и ручку за "
                                                    "50 рублей. Сколько всего? Объясни расчёт."},
                        {"role": "assistant", "content": "Три блокнота: 3 × 60 = 180 рублей. "
                                                         "Вместе с ручкой: 180 + 50 = 230 рублей."},
                        {"role": "user", "content": "Исправляю условие: блокнотов 5, цены прежние. "
                                                    "Сколько теперь? Ответь только числом."}],
              max_completion_tokens=384, autocheck={"type": "numeric", "value": 350},
              expected_values={"initial": 230, "corrected": 350}),
        _task("dialog-corr-pens", "dialog-correction", "short", "auto", "off",
              messages=[{"role": "user", "content": "Купим 4 тетради по 25 рублей и одну ручку "
                                                    "за 20 рублей. Сколько всего? Объясни расчёт."},
                        {"role": "assistant", "content": "Четыре тетради: 4 × 25 = 100 рублей. "
                                                         "Вместе с ручкой: 100 + 20 = 120 рублей."},
                        {"role": "user", "content": "Исправляю условие: тетрадей 6, цены прежние. "
                                                    "Сколько теперь? Ответь только числом."}],
              max_completion_tokens=384, autocheck={"type": "numeric", "value": 170},
              expected_values={"initial": 120, "corrected": 170}),
        _task("retell-garden", "source-retell", "short", "manual", "off",
              prompt="Источник: «В школьном саду посадили 9 кустов смородины. Саженцы привезли "
                     "в мае. Поливать их будут ученики шестого класса». Перескажи источник одним "
                     "предложением, сохранив все три факта и ничего не добавляя.",
              max_completion_tokens=384,
              expected_facts=["В школьном саду посадили 9 кустов смородины.",
                              "Саженцы привезли в мае.",
                              "Поливать будут ученики шестого класса."],
              prohibited_errors=["Изменение числа кустов, месяца или класса.",
                                 "Добавление отсутствующих событий."]),
        _task("retell-trip", "source-retell", "short", "manual", "off",
              prompt="Источник: «В среду класс поехал в музей на автобусе. В музее была "
                     "экскурсия два часа». Перескажи источник одним предложением, сохранив все "
                     "факты и ничего не добавляя.",
              max_completion_tokens=384,
              expected_facts=["В среду класс поехал в музей на автобусе.",
                              "Экскурсия длилась два часа."],
              prohibited_errors=["Замена среды на другой день.",
                                 "Добавление новых событий поездки."]),
        _task("physics-train", "physics", "short", "manual", "off",
              prompt="Поезд проехал 240 километров за 4 часа. Какова его средняя скорость? "
                     "Объясни расчёт и обязательно укажи единицы измерения.",
              max_completion_tokens=384, expected_values={"speed_km_h": 60},
              prohibited_errors=["Деление заменено умножением.",
                                 "Ответ без единиц или с неверными единицами."]),
        _task("physics-bike", "physics", "short", "manual", "off",
              prompt="Велосипедист проехал 36 километров за 2 часа. Какова его средняя скорость? "
                     "Объясни расчёт и обязательно укажи единицы измерения.",
              max_completion_tokens=384, expected_values={"speed_km_h": 18},
              prohibited_errors=["Ответ 72 вместо 18.",
                                 "Ответ без единиц или с неверными единицами."]),
        _task("source-absent-library", "source-absent", "short", "manual", "off",
              prompt="Источник: «Библиотека школы №7 работает с 9:00 до 17:00. По пятницам в "
                     "ней проходит книжный клуб». В каком году открыта библиотека? Ответь "
                     "только на основании источника и кратко объясни вывод.",
              max_completion_tokens=384,
              expected_facts=["В источнике не указан год открытия.",
                              "Определить год по расписанию нельзя."],
              prohibited_errors=["Назван выдуманный год.",
                                 "Уверенное утверждение о годе на основании расписания."]),
        _task("source-absent-captain", "source-absent", "short", "manual", "off",
              prompt="Источник: «Капитан команды — ученик девятого класса. Тренировки проходят "
                     "по вторникам». Сколько лет капитану? Ответь только на основании "
                     "источника и кратко объясни вывод.",
              max_completion_tokens=384,
              expected_facts=["В источнике не указан возраст капитана.",
                              "Класс обучения не равен возрасту."],
              prohibited_errors=["Назван выдуманный возраст.",
                                 "Отождествление номера класса с возрастом."]),
    ]
    return tasks


def _long_dialog_text(doc_index: int, target_tokens: int) -> str:
    """Длинный служебный текст для диалогов (логи/инвентарь)."""
    text, _ = long_document(doc_index, target_tokens)
    return text


def _d_tasks():
    tasks = []
    extra_images = {}
    short_target, long_target = 24_000, 50_000

    # long-retell: документы 1-4 (короткие) и 5-8 (длинные), все manual.
    for doc_index in range(1, 9):
        target = short_target if doc_index <= 4 else long_target
        text, _ = long_document(doc_index, target)
        tasks.append(_task(f"long-retell-{doc_index}", "long-retell", "long", "manual",
                           "off", prompt=text + (
                               "\n\nПерескажи документ в 3–5 предложениях, обязательно упомянув "
                               "температуру, ветер и осадки одной любой записи документа."),
                           max_completion_tokens=768,
                           target_range=[20000, 32000] if target == short_target else [44000, 64000]))

    # long-facts: документы 1-2 (короткие) и 5-6 (длинные), по 3 позиции.
    for doc_index, target in ((1, short_target), (2, short_target),
                              (5, long_target), (6, long_target)):
        for position in ("begin", "mid", "end"):
            prompt, values = fact_question(doc_index, position, target)
            tasks.append(_task(f"long-facts-{doc_index}-{position}", "long-facts",
                               "long", "auto", "off", prompt=prompt,
                               max_completion_tokens=96,
                               target_range=[20000, 32000] if target == short_target else [44000, 64000],
                               autocheck={"type": "json", "value": [
                                   check[1] for check in values["fact_checks"]]},
                               expected_values=values, fact_day=values["day"],
                               fact_position=position))

    # long-tool-dialog: история из «логов инструментов», 2 сценария × 2 длины.
    # Короткий диапазон: 24K + 8K ≈ 32K; длинный: 40K + 16K ≈ 56K.
    for variant, (doc_index, first_target, second_target) in enumerate((
            (3, short_target, 8_000),
            (7, 40_000, 16_000),
            (4, short_target, 8_000),
            (8, 40_000, 16_000))):
        first = _long_dialog_text(doc_index, first_target)
        second = _long_dialog_text(doc_index + 20, second_target)
        if variant < 2:
            first = "Лог проверки серверов:\nСервер ALPHA: ошибка MEM, код 2F.\n" + first
            second = "Второй лог (без ошибок MEM):\n" + second
            final = ("В первом логе найди имя сервера с ошибкой MEM и её код. "
                     "Ответь в формате: ИМЯ КОД.")
            autocheck = {"type": "contains_all", "values": ["ALPHA", "2F"]}
            needle = "ALPHA 2F"
        else:
            first = "Инвентарный список ангара A:\n" + first
            second = "Инвентарный список ангара B:\n" + second.replace("осадки —", "предмет 4471 —", 1)
            final = "В каком ангаре находится предмет с инвентарным номером 4471? Ответь одной буквой."
            autocheck = {"type": "compact", "value": "B"}
            needle = "4471 B"
        tasks.append(_task(f"long-tool-dialog-{variant + 1}", "long-tool-dialog", "long",
                           "auto", "hermes", max_completion_tokens=768,
                           target_range=[20000, 32000] if first_target == short_target else [44000, 64000],
                           historical_tools=[{"type": "function", "function": {
                               "name": "read_log", "description": "Чтение синтетического журнала",
                               "strict": True, "parameters": {
                                   "type": "object", "properties": {
                                       "name": {"type": "string", "enum": ["A", "B"]}},
                                   "required": ["name"], "additionalProperties": False}}}],
                           messages=[{"role": "user", "content": "Проверь оба журнала A и B."},
                                     {"role": "assistant", "content": None, "tool_calls": [{
                                         "id": "call_log_a", "type": "function",
                                         "function": {"name": "read_log", "arguments": '{"name":"A"}'}}]},
                                     {"role": "tool", "tool_call_id": "call_log_a", "content": first},
                                     {"role": "assistant", "content": None, "tool_calls": [{
                                         "id": "call_log_b", "type": "function",
                                         "function": {"name": "read_log", "arguments": '{"name":"B"}'}}]},
                                     {"role": "tool", "tool_call_id": "call_log_b", "content": second},
                                     {"role": "user", "content": final}],
                           autocheck=autocheck, expected_values={"needle": needle}))

    # long-vision-dialog: изображение в середине длинной истории, 2 сценария × 2 длины.
    for variant, (doc_index, target) in enumerate(((4, short_target), (9, long_target),
                                                   (4, short_target), (9, long_target))):
        text, facts = long_document(doc_index, target)
        if variant % 2 == 0:
            image_name = "table-a"
            question = ("На изображении таблица. Какое число находится в строке 2, столбце 3, "
                        "и какая температура была в день 10 документа? Ответь двумя значениями "
                        "через пробел: число таблицы и температура.")
            autocheck = {"type": "numbers_all", "values": [11, facts[10]["temp"]]}
            expected = {"table_cell": "11", "temp_day_10": facts[10]["temp"]}
        else:
            image_name = "formula-a"
            question = ("На изображении формула для X. Чему равен X и какие осадки были в "
                        "день 9 документа? Ответь числом и словом через пробел.")
            autocheck = {"type": "contains_all", "values": ["7", facts[9]["precip"]]}
            expected = {"x": 7, "precip_day_9": facts[9]["precip"]}
        tasks.append(_task(f"long-vision-dialog-{variant + 1}", "long-vision-dialog", "long",
                           "auto", "hermes", max_completion_tokens=768,
                           target_range=[20000, 32000] if target == short_target else [44000, 64000],
                           messages=[{"role": "user", "content": [
                               image(image_name) if variant % 2 == 0 else image(image_name),
                               {"type": "text", "text": text}]},
                               {"role": "assistant", "content": "Изображение и документ получены."},
                               {"role": "user", "content": question}],
                           images=[image_name], autocheck=autocheck,
                           expected_values=expected))
        extra_images.setdefault(image_name, True)

    return tasks, extra_images


def new_tasks() -> list:
    """84 новые задачи набора (детерминированный порядок)."""
    ocr_text, _ = _ocr_text_tasks()
    return sorted(
        ocr_text + _table_tasks() + _receipt_tasks() + _formula_tasks()
        + _multi_image_tasks() + _big_text_tasks() + _code_tasks()
        + _tool_chain_tasks() + _json_schema_tasks() + _hermes_tasks()
        + _a_new_tasks() + _d_tasks()[0],
        key=lambda task: (task["domain"], task["id"]))


def image_blocks() -> dict:
    """Все изображения набора: имя -> блок image_url (детерминированно)."""
    _, d_images = _d_tasks()
    blocks = dict(d_images)
    blocks.update({name: image(name) for name in image_names()})
    _, ocr_images = _ocr_text_tasks()
    blocks.update(ocr_images)
    return blocks


def all_tasks() -> list:
    """Читать закреплённые задачи из обоих JSONL, не генерировать их заново."""
    quality = Path(__file__).with_name("faex2_quality.jsonl")
    existing = [json.loads(line) for line in quality.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    for task in existing:
        task.setdefault("bucket", "short")
        task.setdefault("thinking", "off")
        task["evaluation"] = "manual"
        # 12 существующих задач входят в домен A; исходные themes сохраняются в "domains".
        task.setdefault("domain", "family-chat")
    registry = Path(__file__).with_name("faex2_appset.jsonl")
    new = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    return existing + new


def export(path: Path) -> None:
    rows = new_tasks()
    path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in rows),
                    encoding="utf-8")


def main() -> None:
    export(Path(__file__).with_name("faex2_appset.jsonl"))
    counts = {}
    for task in new_tasks():
        counts[task["domain"]] = counts.get(task["domain"], 0) + 1
    for domain in sorted(counts):
        print(f"{domain}: {counts[domain]}")


if __name__ == "__main__":
    main()
