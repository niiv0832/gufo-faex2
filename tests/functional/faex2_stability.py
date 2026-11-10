"""C-клиент: контрольные истории и ограниченный soak поверх acceptance.Runner.

run_control(runner, records, groups=None) использует подтверждённое B3 число
архивных записей. groups: short, vision, batch, long_cache; порядок фиксирован,
long_cache всегда исполняется целиком. Перед выбранными группами всегда идёт
обязательная mtp_c1: один sampled JSON с 40 квадратами, proposed>0/accepted>0.
Он входит в cases/history и тайминги, одинаков для full и targeted повторов.
Control v1 без этого доказательства не принимается compare_reports.
Каждый повтор требует нового Runner и
свежего отдельного server cache. Вход comparison_conditions в runner.report:
model_sha256/mmproj_sha256/harness_sha256/runtime_sha256, точный profile
(sampler/cache/think и остальные настройки без уникальных run paths/порта), cache_start='fresh-separated',
host_conditions (manifest условий стенда). Source/binary SHA сохраняются отдельно
в server_record и могут отличаться. Этот модуль сам стенд/сервер не проверяет.

compare_reports(base, candidate) принимает завершённые report dict или пути
report.json. Сравнивает одинаковый набор выбранных групп, все запросы и пять фаз
по отдельности. NEEDS_INVESTIGATION — повод для повторов, не доказанная регрессия.
Нулевой prefill допускает явно отмеченный N/A; ненулевая работа требует метрику.

run_soak(runner, warmup_s=600, duration_s=1800) не позволяет сократить эти фазы;
их суммарный заданный срок <=6000 с, плюс завершение текущего HTTP-batch <=120 с
(Runner имеет собственную надбавку watchdog). clock заменяется только CPU-тестами.
Один reasoning off/on + red-vision цикл входит в прогрев; измерение повторяет
фиксированные C1/C2/C4 истории с temperature=1. Никаких новых бесед/кэшей по циклам.
Владелец обязан отдельно вести серверный мониторинг каждые 2 с и квалифицировать
RSS/GTT/MemAvailable/температуру/kernel logs по сохранённым временным границам.
WORKLOAD_PASS_REQUIRES_MONITOR_REVIEW не означает полную приёмку стабильности.
Оба run_* сохраняют report после каждого шага; finish() Runner вызывает владелец.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

from faex2 import require
from faex2_acceptance import grade
import faex2_family as family
from faex2_images import image_content
from metrics import timing_measurement

GROUPS = ("short", "vision", "batch", "long_cache")
PHASES = ("prefill_ms", "decode_ms", "ttft_ms", "client_ttfa_ms", "wall_ms")
COUNTERS = ("prompt_tokens", "prefill_tokens", "cached_tokens", "completion_tokens")


def _mtp_c1():
    squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324,
               361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1089,
               1156, 1225, 1296, 1369, 1444, 1521, 1600]
    request = family._chat([
        {"role": "system", "content": "Независимая контрольная беседа C1_MTP_742."},
        {"role": "user", "content": "Вычисли квадраты целых чисел от 1 до 40 включительно. "
         "Верни только JSON: session — идентификатор этой беседы, squares — сорок квадратов по порядку."}],
        family._schema({"session": {"type": "string"}, "squares": {"type": "array",
            "items": {"type": "integer"}, "minItems": 40, "maxItems": 40}}),
        temperature=1, cache=False, max_tokens=384)
    return "control-mtp-c1", request, {"json": {"session": "C1_MTP_742", "squares": squares}, "draft_work": True}


def _conditions(report):
    value = report.get("comparison_conditions")
    require(isinstance(value, dict), "comparison_conditions are required")
    for name in ("model_sha256", "mmproj_sha256", "harness_sha256", "runtime_sha256"):
        require(isinstance(value.get(name), str) and re.fullmatch("[0-9a-f]{64}", value[name]),
                "missing/invalid comparison condition: " + name)
    profile = value.get("profile")
    require(isinstance(profile, dict) and profile.get("mode") == "mtp" and
            profile.get("sessions") == 4 and profile.get("context") == 131072 and
            profile.get("vision") is True and isinstance(profile.get("sampler"), dict),
            "comparison requires the matched MTP/vision family profile")
    require(value.get("cache_start") == "fresh-separated" and
            isinstance(value.get("host_conditions"), dict) and value["host_conditions"],
            "fresh separate cache and host conditions must be recorded")
    return value


def _batch_cases(prefix, width, cache=False):
    schema = family._schema({"session": {"type": "string"}, "products": {
        "type": "array", "items": {"type": "integer"}, "minItems": 8, "maxItems": 8}})
    cases = []
    for marker, factor, products in family.SESSIONS[:width]:
        request = family._chat([
            {"role": "system", "content": "Независимая семейная беседа " + marker},
            {"role": "user", "content": f"Вычисли произведения числа {factor} на числа от 20 до 27 "
             "включительно. Верни JSON: session — идентификатор этой беседы, products — восемь "
             "произведений по порядку. Никаких пояснений."}], schema,
            temperature=1, cache=cache, max_tokens=256)
        expected = {"json": {"session": marker, "products": products},
                    "excludes": [other for other, _, _ in family.SESSIONS if other != marker]}
        if width == 1: expected["draft_work"] = True
        cases.append((f"{prefix}-c{width}-{marker}", request, expected))
    return cases


def _vision(identifier):
    request = family._chat([{"role": "user", "content": [image_content("red"),
        {"type": "text", "text": "Назови основной цвет изображения. Ответь только словом red, "
         "blue или green на английском языке, строчными буквами."}]}],
        temperature=1, cache=False, max_tokens=32)
    return identifier, request, {"text": "red", "fresh_prefill": True}


def run_control(runner, records, groups=None):
    """Одинаковые inputs и group history; отбор группы не отрезает cache-прелюдию."""
    _conditions(runner.report)
    require(runner.mode == "mtp", "control requires MTP")
    chosen = GROUPS if groups is None else tuple(groups)
    require(chosen and len(set(chosen)) == len(chosen) and set(chosen) <= set(GROUPS),
            "unknown/empty/duplicate control groups")
    chosen = tuple(group for group in GROUPS if group in chosen)
    cold_request = family._long_request(records, cache=False)  # Проверить глубину до первого HTTP.
    require("control" not in runner.report, "control was already started in this Runner")
    control = {"version": 2, "status": "RUNNING", "groups": ["mtp_c1", *chosen],
               "records": records, "cases": [], "started_at": time.time()}
    runner.report["control"] = control
    histories = {group: [] for group in ("mtp_c1", *chosen)}
    runner.save()

    def dispatch(group, cases, *, batch=False, timeout=120):
        inputs = [family._hash(request) for _, request, _ in cases]
        histories[group].append(sorted(inputs))
        history = family._hash(histories[group])
        for (identifier, _, expected), fingerprint in zip(cases, inputs):
            control["cases"].append({"id": identifier, "group": group, "width": len(cases),
                "request_sha256": fingerprint, "expected_sha256": family._hash(expected),
                "history_sha256": history, "timeout": timeout})
        runner.save()
        results = runner.run_batch(cases, timeout=timeout) if batch else [
            runner.run_case(*cases[0], timeout=timeout)]
        for result in results: family._work(result)
        return results[0] if not batch else results

    try:
        dispatch("mtp_c1", [_mtp_c1()])
        if "short" in chosen:
            for streaming, suffix in ((False, "json"), (True, "sse")):
                request = family._chat([{"role": "user", "content":
                    "Вычисли 17 × 23. Верни только JSON с целым полем answer."}],
                    family._schema({"answer": {"type": "integer"}}),
                    temperature=1, cache=False, max_tokens=64)
                request["body"]["stream"] = streaming
                if not streaming: del request["body"]["stream_options"]
                dispatch("short", [("control-short-" + suffix, request, {"json": {"answer": 391}})])
        if "vision" in chosen:
            dispatch("vision", [_vision("control-vision-red")])
        if "batch" in chosen:
            for width in (2, 4):
                dispatch("batch", _batch_cases("control-batch", width), batch=True)
        if "long_cache" in chosen:
            def long(identifier, request, expected):
                return dispatch("long_cache", [("control-long-" + identifier, request, expected)], timeout=600)

            cold = long("cold", cold_request, family._long_expected())
            depth, cached, _ = family._work(cold)
            require(cached == 0, "cold control unexpectedly reused cache")
            repeated = family._long_request(records)
            long("repeat", repeated, {**family._long_expected(), "min_cached_tokens": depth - 16,
                                      "max_prefill_tokens": 16})
            continued = deepcopy(repeated)
            continued["body"]["messages"] += [
                {"role": "assistant", "content": json.dumps(family.KEYS, ensure_ascii=False, separators=(",", ":"))},
                {"role": "user", "content": "Повтори те же три ключа в том же JSON-формате."}]
            long("continue", continued, {**family._long_expected(), "min_cached_tokens": depth - 16,
                                         "max_prefill_tokens": 256})
            # Полная стоимость первой развилки входит в cases/history и comparison.
            warm_request, warm_expected = family._branch_warmup(records)
            long("branch-warmup", warm_request, warm_expected)
            edited = family._long_request(records, edited=True)
            changed = long("edit", edited, {**family._long_expected(True), "min_cached_tokens": depth - 256,
                                            "max_prefill_tokens": 256})
            edited_depth, _, _ = family._work(changed)
            long("edit-repeat", edited, {**family._long_expected(True),
                "min_cached_tokens": edited_depth - 16, "max_prefill_tokens": 16})
        control["status"] = "PASS"
    except BaseException as error:
        control.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        runner.report["status"] = "FAIL"
        raise
    finally:
        control["finished_at"] = time.time()
        runner.save()
    return runner.report


def _report(value):
    result = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    require(isinstance(result, dict) and result.get("status") == "PASS" and
            result.get("control", {}).get("status") == "PASS", "control report is unfinished or failed")
    _conditions(result)
    control = result["control"]
    require(control.get("version") == 2 and control.get("groups", [])[:1] == ["mtp_c1"],
            "control lacks mandatory explicit MTP C1 qualification")
    identifier, request, expected = _mtp_c1()
    fingerprint = family._hash(request)
    descriptor = {"id": identifier, "group": "mtp_c1", "width": 1, "request_sha256": fingerprint,
                  "expected_sha256": family._hash(expected), "history_sha256": family._hash([[fingerprint]]), "timeout": 120}
    require(control.get("cases", [])[:1] == [descriptor], "MTP C1 input/history changed or is missing")
    gate = [row for row in result.get("requests", []) if row.get("id") == identifier]
    require(len(gate) == 1 and gate[0].get("status") == "PASS", "MTP C1 result is missing or failed")
    grade(gate[0].get("result", {}), expected, "mtp")
    return result


# Место допуска сдвигает только первый токен. Полное время запроса от места почти не
# зависит (пакет заканчивается вместе), поэтому wall_ms всегда сравнивается по имени:
# иначе замедление после первого токена было бы скрыто при перестановке.
ADMISSION_PHASES = ("ttft_ms", "client_ttfa_ms")


def _batches(cases):
    """Параллельные группы: запросы одного dispatch имеют общую history и width>1."""
    batches = {}
    for case in cases:
        if case.get("width", 1) > 1:
            batches.setdefault((case["group"], case["width"], case["history_sha256"]), []).append(case["id"])
    return [names for names in batches.values() if len(names) > 1]


def _batch_order(names, source):
    """Места допуска 1…N либо None, если сервер не сообщил однозначный порядок."""
    values = [source[name]["result"].get("metrics", {}).get("resident_requests_at_admission")
              for name in names]
    if sorted(value for value in values if type(value) is int) != list(range(1, len(names) + 1)):
        return None
    return dict(zip(names, values))


def _admission_ranks(cases, sources):
    """Запросы, чьё место допуска в параллельной группе различается между прогонами.

    Клиент отправляет группу отдельными процессами, поэтому порядок допуска — гонка.
    Время до первого токена определяется местом, а не именем запроса. Сравнение по
    имени при перестановке сопоставляет разные условия; такие ячейки помечаются
    UNMATCHED_ADMISSION и заменяются сравнением одинаковых мест. Без однозначных
    мест с обеих сторон прежнее строгое сравнение по имени сохраняется.
    """
    moved = {}
    for names in _batches(cases):
        orders = [_batch_order(names, source) for source in sources]
        if None in orders or orders[0] == orders[1]:
            continue
        for name in names:
            moved[name] = (orders[0][name], orders[1][name])
    return moved


def _rank_measurements(cases, sources, moved):
    rows = []
    for names in _batches(cases):
        if not any(name in moved for name in names):
            continue
        group = next(case["group"] for case in cases if case["id"] == names[0])
        label = names[0].rsplit("-", 1)[0]
        for rank in range(1, len(names) + 1):
            pair = [next(name for name in names if moved[name][side] == rank) for side in (0, 1)]
            results = [source[name]["result"] for source, name in zip(sources, pair)]
            metrics = [result.get("metrics", {}) for result in results]
            for phase in ADMISSION_PHASES:
                values = [metric.get(phase) for metric in metrics]
                require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                            for value in values), f"{label}: invalid {phase} at admission {rank}")
                row = timing_measurement(f"{label}#admission-{rank}", phase, *values,
                                         request_key=[group, *pair])
                prompt_work = all(metrics[0][field] == metrics[1][field]
                                  for field in ("prompt_tokens", "prefill_tokens", "cached_tokens"))
                row["status"] = ("UNMATCHED_WORK" if not prompt_work else
                                 "NEEDS_INVESTIGATION" if row["exceeds_margin"] else "PASS")
                row["paired_requests"] = pair
                rows.append(row)
    return rows


def compare_reports(base, candidate):
    """Никаких средних или утверждений о регрессии из единственного наблюдения."""
    base, candidate = _report(base), _report(candidate)
    require(base["comparison_conditions"] == candidate["comparison_conditions"], "unmatched comparison conditions")
    require(base["control"]["groups"] == candidate["control"]["groups"] and
            base["control"]["records"] == candidate["control"]["records"] and
            base["control"]["cases"] == candidate["control"]["cases"],
            "unmatched requests or group cache history")
    cases = base["control"]["cases"]
    require(cases and len({case["id"] for case in cases}) == len(cases), "missing/duplicate control requests")
    sources = []
    for report in (base, candidate):
        rows = report.get("requests", [])
        require(len({row["id"] for row in rows}) == len(rows), "duplicate request results")
        index = {row["id"]: row for row in rows}
        require(all(case["id"] in index for case in cases), "missing request result")
        sources.append(index)
    measurements, changes, output_changes = [], [], []
    ranks = _admission_ranks(cases, sources)
    for case in cases:
        name = case["id"]
        rows = [source[name] for source in sources]
        require(all(row.get("status") == "PASS" and row.get("result", {}).get("status") == "PASS"
                    for row in rows), "failed control quality: " + name)
        results = [row["result"] for row in rows]
        metrics = [result.get("metrics", {}) for result in results]
        for result in results: family._work(result)
        require(all(type(metric.get("completion_tokens")) is int and metric["completion_tokens"] >= 0
                    for metric in metrics), "invalid completion token count")
        different = [field for field in COUNTERS if metrics[0][field] != metrics[1][field]]
        if different: changes.append({"case": name, "fields": different,
            "baseline": {field: metrics[0][field] for field in COUNTERS},
            "candidate": {field: metrics[1][field] for field in COUNTERS}})
        hashes = [result.get("output_sha256") for result in results]
        require(all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in hashes),
                "missing output fingerprint")
        if hashes[0] != hashes[1]: output_changes.append(name)
        for phase in PHASES:
            values = [result.get(phase) if phase == "wall_ms" else metric.get(phase)
                      for result, metric in zip(results, metrics)]
            work = "prefill_tokens" if phase == "prefill_ms" else "completion_tokens"
            zero_work = phase in ("prefill_ms", "decode_ms") and all(metric[work] == 0 for metric in metrics)
            require(all(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for value in values), f"{name}: invalid {phase}")
            if any(value is None for value in values):
                require(zero_work, f"{name}: missing {phase} despite measured work")
                measurements.append({"case": name, "metric": phase, "status": "N/A",
                                     "reason": "zero " + work, "baseline": values[0], "candidate": values[1]})
                continue
            row = timing_measurement(name, phase, *values, request_key=[case["group"], name])
            row["status"] = ("UNMATCHED_WORK" if different else
                             "NEEDS_INVESTIGATION" if row["exceeds_margin"] else "PASS")
            moved = ranks.get(name)
            if moved and phase in ADMISSION_PHASES and not different:
                # Один и тот же запрос занял разные места допуска: условия не совпали.
                row.update(status="UNMATCHED_ADMISSION", baseline_admission_rank=moved[0],
                           candidate_admission_rank=moved[1])
            measurements.append(row)
    measurements.extend(_rank_measurements(cases, sources, ranks))
    flagged = any(row["status"] in ("NEEDS_INVESTIGATION", "UNMATCHED_WORK") and "#admission-" in row["case"]
                  or row["status"] == "NEEDS_INVESTIGATION" for row in measurements)
    return {"status": "NEEDS_INVESTIGATION" if flagged or changes or output_changes else "PASS",
            "slowdown_tolerance": .05, "noise_ms": 3, "measurements": measurements,
            "work_changes": changes, "output_changes": output_changes,
            "interpretation": "one matched observation; flags require affected-history follow-up"}


def run_soak(runner, warmup_s=600, duration_s=1800, *, clock=time):
    """Фиксированные истории; окончание фазы — после последнего начатого batch."""
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (warmup_s, duration_s))
            and warmup_s >= 600 and duration_s >= 1800 and warmup_s + duration_s <= 6000,
            "soak requires >=600 s warmup, >=1800 s measurement, total <=6000 s")
    require(runner.mode == "mtp" and "stability" not in runner.report, "soak already started or wrong mode")
    state = {"version": 1, "status": "RUNNING", "warmup_s": warmup_s, "duration_s": duration_s,
             "cycle": [1, 2, 4], "monitor_interval_s": 2, "external_monitor_review": "REQUIRED",
             "completed_batches": 0}
    runner.report["stability"] = state
    runner.save()
    try:
        for phase, duration in (("warmup", warmup_s), ("measurement", duration_s)):
            start = clock.monotonic()
            window = {"started_at": clock.time(), "planned_duration_s": duration,
                      "first_request_index": len(runner.report["requests"])}
            state[phase] = window
            runner.save()
            if phase == "warmup":
                family._reasoning(SimpleNamespace(case=runner.run_case))
                runner.run_case(*_vision("soak-vision-red"))
            step = 0
            while clock.monotonic() - start < duration:
                width = (1, 2, 4)[step % 3]
                before = clock.monotonic()
                cases = _batch_cases(f"soak-{phase}-{step:05d}", width, cache=True)
                results = runner.run_batch(cases, timeout=120)
                for result in results: family._work(result)
                require(clock.monotonic() > before, "monotonic clock did not advance")
                step += 1
                state["completed_batches"] += 1
                window["completed_batches"] = step
                window["elapsed_s"] = clock.monotonic() - start
                runner.save()
            window.update(finished_at=clock.time(), elapsed_s=clock.monotonic() - start,
                          last_request_index=len(runner.report["requests"]) - 1)
            runner.save()
        state["status"] = "WORKLOAD_PASS_REQUIRES_MONITOR_REVIEW"
    except BaseException as error:
        state.update(status="FAIL", failed_at=clock.time(), error=type(error).__name__ + ": " + str(error))
        runner.report["status"] = "FAIL"
        raise
    finally:
        runner.save()
    return runner.report
