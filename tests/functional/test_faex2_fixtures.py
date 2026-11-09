"""Проверки публичных фикстур FAEX2 без модели, сети и сторонних библиотек."""

import base64
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))


def decode_png(content):
    """Независимый узкий декодер PNG RGB8: проверяет контейнер, CRC и пиксели."""
    if set(content) != {"type", "image_url"} or content["type"] != "image_url":
        raise AssertionError("Ожидался блок OpenAI Chat image_url")
    url = content["image_url"]["url"]
    prefix = "data:image/png;base64,"
    if not url.startswith(prefix):
        raise AssertionError("Ожидался PNG data URL")
    png = base64.b64decode(url[len(prefix):], validate=True)
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("Неверная сигнатура PNG")
    offset, chunks = 8, []
    while offset < len(png):
        size = int.from_bytes(png[offset:offset + 4], "big")
        kind = png[offset + 4:offset + 8]
        payload = png[offset + 8:offset + 8 + size]
        checksum = int.from_bytes(png[offset + 8 + size:offset + 12 + size], "big")
        if len(payload) != size or checksum != zlib.crc32(kind + payload):
            raise AssertionError("Повреждённый PNG chunk")
        chunks.append((kind, payload))
        offset += 12 + size
    if offset != len(png) or [item[0] for item in chunks] != [b"IHDR", b"IDAT", b"IEND"]:
        raise AssertionError("Неверная последовательность PNG chunks")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", chunks[0][1])
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise AssertionError("Ожидался несмешанный RGB8 PNG")
    raw = zlib.decompress(chunks[1][1])
    stride = width * 3 + 1
    if len(raw) != height * stride or any(raw[row * stride] for row in range(height)):
        raise AssertionError("Неверные размеры или фильтр строк PNG")
    pixels = b"".join(raw[row * stride + 1:(row + 1) * stride] for row in range(height))
    return width, height, pixels


class ImageFixtureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("faex2_images"),
                             "Генератор публичных изображений ещё не реализован")
        self.images = importlib.import_module("faex2_images")

    def test_swatch_has_exact_rgb_for_every_pixel(self):
        # Перепутанный канал, размер или незакрашенная строка ломают проверку.
        for name, rgb in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
            with self.subTest(color=name):
                width, height, pixels = decode_png(self.images.image_content(name))
                self.assertEqual((width, height), (128, 128))
                self.assertEqual(pixels, bytes(rgb) * 128 * 128)

    def test_unknown_swatch_is_rejected_instead_of_silently_changing_color(self):
        with self.assertRaises(ValueError):
            self.images.image_content("purple")

    def test_text_has_independently_checked_glyph_rows_and_clear_margins(self):
        width, height, pixels = decode_png(self.images.text_image_content())
        self.assertEqual((width, height), (640, 192))
        self.assertEqual(pixels[:54 * width * 3], b"\xff" * (54 * width * 3))
        self.assertEqual(pixels[138 * width * 3:], b"\xff" * (54 * width * 3))
        # Ручная спецификация чёрных интервалов трёх строк GUFO 742.
        # Координаты относятся к клеткам текста; это не копия словаря шрифта.
        golden_rows = {
            60: [(1, 4), (6, 7), (10, 11), (12, 17), (19, 22),
                 (30, 35), (39, 40), (43, 46)],
            96: [(0, 1), (2, 5), (6, 7), (10, 11), (12, 16), (18, 19),
                 (22, 23), (32, 33), (36, 37), (39, 40), (45, 46)],
            132: [(1, 4), (7, 10), (12, 13), (19, 22), (31, 32),
                  (39, 40), (42, 47)],
        }
        for y, spans in golden_rows.items():
            expected_black = {x for start, stop in spans
                              for x in range(38 + start * 12, 38 + stop * 12)}
            row = pixels[y * width * 3:(y + 1) * width * 3]
            actual_black = {x for x in range(width) if row[x * 3:x * 3 + 3] == b"\0\0\0"}
            with self.subTest(row=y):
                self.assertEqual(actual_black, expected_black)
        # SHA заранее вычислен из независимой построчной спецификации всех 7 строк.
        # Хешируем пиксели, чтобы версия zlib не меняла смысл эталона.
        self.assertEqual(hashlib.sha256(pixels).hexdigest(),
                         "55056333621b13a0ea93250431999f4bc8e19714b8fabe42286de2623a816f50")

    def test_text_generation_is_repeatable_and_only_black_on_white(self):
        first = self.images.text_image_content()
        self.assertEqual(first, self.images.text_image_content())
        _, _, pixels = decode_png(first)
        self.assertEqual({pixels[i:i + 3] for i in range(0, len(pixels), 3)},
                         {b"\0\0\0", b"\xff\xff\xff"})

    def test_existing_jpeg_and_webp_fixtures_work_with_the_small_generator(self):
        from image_inputs import image_cases, invalid_image_cases
        formats = dict(image_cases(self.images.image_content))
        self.assertTrue({"png", "jpeg", "webp_lossless", "webp_lossy"} <= formats.keys())
        jpeg = base64.b64decode(formats["jpeg"].split(",", 1)[1], validate=True)
        self.assertTrue(jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"))
        for name, chunk in (("webp_lossless", b"VP8L"), ("webp_lossy", b"VP8 ")):
            webp = base64.b64decode(formats[name].split(",", 1)[1], validate=True)
            self.assertEqual((webp[:4], webp[8:12], webp[12:16]), (b"RIFF", b"WEBP", chunk))
        bad_url = next(url for name, url, _ in invalid_image_cases(self.images.image_content)
                       if name == "bad_base64")
        with self.assertRaises(ValueError):
            base64.b64decode(bad_url.split(",", 1)[1], validate=True)


class QualityFixtureTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("faex2_quality.jsonl")
        self.assertTrue(path.is_file(), "Школьные вопросы и goldens ещё не закреплены")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines and all(line.strip() for line in lines))
        self.cases = [json.loads(line) for line in lines]
        self.by_id = {case["id"]: case for case in self.cases}

    def test_case_ids_and_request_shape_are_safe_to_consume(self):
        self.assertEqual(len(self.by_id), len(self.cases), "Повторный ID затрёт результат")
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertRegex(case["id"], r"^[a-z][a-z0-9-]+$")
                self.assertIsInstance(case["prompt"], str)
                self.assertRegex(case["prompt"], r"[А-Яа-яЁё]")
                self.assertNotIn("json", case["prompt"].lower())
                self.assertIs(type(case["max_completion_tokens"]), int)
                self.assertGreaterEqual(case["max_completion_tokens"], 384)
                self.assertLessEqual(case["max_completion_tokens"], 768)
                for key in ("expected_facts", "prohibited_errors", "domains"):
                    self.assertIsInstance(case[key], list)
                    self.assertTrue(case[key] and all(isinstance(value, str) and value.strip()
                                                     for value in case[key]))

    def test_required_domains_cannot_disappear_from_the_quality_run(self):
        mandatory = {"fractions", "linear-equation", "percentages", "proportions", "units",
                     "error-explanation", "summary", "source-grounding", "russian-grammar",
                     "seasons", "photosynthesis", "logic", "dialog-correction"}
        covered = {domain for case in self.cases for domain in case["domains"]}
        self.assertFalse(mandatory - covered, f"Нет обязательных тем: {mandatory - covered}")

    def test_every_answer_requires_human_scores_and_critical_failures_override_sum(self):
        dimensions = {"correctness", "instruction", "russian_clarity", "explanation"}
        critical = {"incorrect_math", "incorrect_fact", "fabricated_source", "empty", "truncated"}
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["evaluation"], "manual")
                self.assertEqual(set(case["rubric"]), dimensions)
                for anchors in case["rubric"].values():
                    self.assertEqual(set(anchors), {"0", "1", "2"})
                    self.assertTrue(all(isinstance(text, str) and text.strip()
                                        for text in anchors.values()))
                self.assertTrue(critical <= set(case["critical_failures"]))
                self.assertEqual(case["pass_criteria"], {
                    "correctness": 2, "minimum_per_dimension": 1, "minimum_total": 6,
                    "critical_failure_result": "FAIL"})
                self.assertEqual(case["review_record"], ["scores", "rationale", "critical_failures"])

    def test_math_goldens_are_independently_recomputed_before_inference(self):
        # Арифметика теста не использует ни ответы модели, ни код генератора requests.
        expected = {
            "fractions": {"answer": str(Fraction(3, 4) + Fraction(2, 3))},
            "linear-equation": {"x": 15 // 3 + 2},
            "discount": {"price_rub": 1200 * (100 - 15) // 100,
                         "saved_rub": 1200 * 15 // 100},
            "speed": {"speed_km_h": 150 // 3},
            "fraction-error": {"answer": str(Fraction(1, 2) + Fraction(1, 3))},
            "dialog-correction": {"initial_rub": 4 * 45 + 30, "corrected_rub": 5 * 45 + 30},
        }
        for name, values in expected.items():
            with self.subTest(case=name):
                self.assertEqual(self.by_id[name]["expected_values"], values)
                facts = " ".join(self.by_id[name]["expected_facts"])
                for value in values.values():
                    self.assertRegex(facts, r"(?<!\d)" + re.escape(str(value)) + r"(?!\d)",
                                     "Числовой golden расходится с фактами для ревью")

    def test_dialog_replays_the_old_answer_then_corrects_only_the_notebook_count(self):
        case = self.by_id["dialog-correction"]
        messages = case["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
        self.assertRegex(messages[0]["content"], r"4 тетради.*45.*30")
        self.assertIn("210", messages[1]["content"])
        self.assertEqual(messages[-1]["content"], case["prompt"])
        self.assertRegex(messages[-1]["content"], r"5 тетрад")


if __name__ == "__main__":
    unittest.main()
