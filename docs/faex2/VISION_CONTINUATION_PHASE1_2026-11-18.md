# Фазы 1–2 расследования падения `model-vision-serving-mtp` (Этап 1, Б1)

Дата: 2026-11-18 (уточнено повторным замером **2026-11-19**). Статус: **завершено (диагноз + точные построчные измерения `maxΔ` и `KL` по общему префиксу `0..3`), тест `35/36` не ослаблялся и по решению владельца остаётся в связке с контрактом `Q1`**.
Задача: объяснить стабильное падение `qwen_vision_serving_test` на проверке
`continued image cache differs from cold prefill` ([vision_serving_test.cpp:982-984](../../tests/models/qwen/vision_serving_test.cpp#L982-L984))
в наборе `r3-tests` (35/36) и решить, является ли оно дефектом логики или следствием документированной численной неэквивалентности путей.
Продакшен `b81812aa` не изменялся.

## 1. Что видно из сохранённого артефакта падения

Источник: `.local/remed-20261113/results/final-tests/model-vision-serving-mtp.out` (код завершения 1, `mtp-embedded`, 2026-11-13). Ключевые строки:

1. **Все оракулы состояния — точные:**
   - `append_max_logit_error=0.00` — все 8 сравнений оракула добавления/восстановления снимка ([CheckAppendSnapshotOracle](../../tests/models/qwen/vision_serving_test.cpp#L450-L590));
   - `Qwen fresh live-history oracle exact=1` — трижды (живая история, диск AR, диск MTP) ([CheckQwenIncrementalOracle](../../tests/models/qwen/vision_serving_test.cpp#L263-L311), строки 128/133/140 лога);
   - все дисковые восстановления `exact=1`, `multiple images: Red, Blue`.
2. **Проверка инвариантности границ чанков в этом же прогоне прошла.** Внутри `CheckQwenIncrementalOracle` есть требование побитного равенства логитов двухчанкового и одночанкового матричного prefill того же промпта ([vision_serving_test.cpp:297-310](../../tests/models/qwen/vision_serving_test.cpp#L297-L310)). Раз тест дошёл до финальной строки `984`, эта проверка **пройдена**: для данного промпта (135 токенов, красное изображение, `previous.prompt_tokens = 94`) форма чанков матричного prefill логиты не меняет.
3. **Само расхождение — один токен на почти равных кандидатах:**
   ```text
   continued cache: I saw red, as the entire image is a solid, uniform field of red
   continued cold:  I saw red — the entire image is a solid, uniform field of red with
   cached IDs: 40 5312 2438 11   430 279 4307 2099 369 264 6235 11 ...
   cold IDs:   40 5312 2438 1892 279 4307 2099 369 264 6235 11 ...
   ```
   Первые три токена совпадают; на 4-м шаге жадного выбора кэш-путь берёт `,` (11), холодный — `—` (1892), дальше тексты почти ре-синхронизируются. Это подпись **численного шума на почти равных логитах**, а не порчи состояния: при ошибке позиций/эмбеддингов/состояния расхождение начиналось бы с первого токена и каскадно нарастало.

Важно: `--append-only` завершается сразу после `CheckAppendSnapshotOracle` ([vision_serving_test.cpp:686-688](../../tests/models/qwen/vision_serving_test.cpp#L686-L688)) и **вообще не исполняет** падающую проверку — поэтому его PASS с `append_max_logit_error=0.00` (подтверждённый и на `b81812aa` в `LLM_FAEX2/results/2026-11-18-t2a2-deploy/out/06_vision_serving_test.log`) ничего не говорит о ней.

## 2. Механизм расхождения (по коду и логам)

**Кэш-путь `continued`:** ответ ассистента на красное изображение генерировался декодированием, поэтому KV-записи для токенов 94–109 (текст ответа) записаны **decode-ядрами** (`ForwardToken`); при продолжении из кэша до-префиллится только суффикс (~25 токенов) — `live_checkpoint tokens=110`, `prefix=110 total=135` в логе.

**Холодный путь `cold_continuation`:** свежий бэкенд префиллит все 135 токенов **матричным prefill** (`ForwardPromptBatch`), включая тот же текст ответа.

То есть два пути отличаются **арифметикой записи одних и тех же токенов в KV** (decode vs prefill), а не размером чанка: чанк-инвариантность матричного prefill для этого промпта доказана побитно (п. 1.2).

Это **документированное свойство модели**, а не новая находка:

- [INVESTIGATION_RESULT.md](INVESTIGATION_RESULT.md) §В: «три пути — prefill, декодирование, пакетная проверка — считают немного по-разному»; логиты строк проверки отличаются от декодирования до 0,82; «потребовало бы побитного согласования ядер экспертов» — сознательно не исправлялось.
- Сам тест это признаёт в комментарии ([vision_serving_test.cpp:276-278](../../tests/models/qwen/vision_serving_test.cpp#L276-L278)): «the retained reply used decode arithmetic, not matrix prefill. The separate cold comparison below deliberately continues to check that **stronger guarantee**» — т.е. проверка `continued == cold` требует более сильной гарантии, чем даёт текущая численная база MoE.
- [AUDIT_CLOSURE_ROUND3_2026-11-14.md](AUDIT_CLOSURE_ROUND3_2026-11-14.md) №9/№12 и расчёт стоимости этапа 8 (`round3/stage8/README.md`): шум самой арифметики (0,003–0,005) больше допуска черновика (0,001) — тот же класс явления; падающая проверка «не ослаблялась», статус «ждёт решения владельца о контракте» ([ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md) §2.А.1–2, контракт `Q1` DRAFT).

## 3. Аудит кода пути `vision + continuation cache` — дефектов не найдено

- **M-RoPE позиции** детерминированы структурой промпта: `RopeLayout::Position` вычисляется из сеток изображений ([prompt.cpp:104-121](../../src/models/qwen/vision/prompt.cpp#L104-L121)), суффикс за пределами изображений использует `physical + delta` ([rope.hpp:15-28](../../src/models/qwen/vision/rope.hpp#L15-L28)); `Prepare` строит токены/позиции одинаково для обоих путей ([prompt.cpp:249-353](../../src/models/qwen/vision/prompt.cpp#L249-L353)). Позиции не зависят от истории кэша.
- **Подготовка и дочитывание в serving-слое:** `PrepareQwenPrompt` ([inference_backend.cpp:122-154](../../src/cli/serve/inference_backend.cpp#L122-L154)) передаёт `QwenImageContext` и `Prompt`; при продолжении `RestoreOrFork` ([inference_backend.cpp:1517-1542](../../src/cli/serve/inference_backend.cpp#L1517-L1542)) переносит состояние исполнителя, позицию, фронтир и логиты, а `QwenTextRunner::Prefill` ([inference_backend.cpp:1300-1360](../../src/cli/serve/inference_backend.cpp#L1300-L1360)) до-префиллит суффикс через `ForwardPromptBatch` с тем же `offset`.
- **Снимок сохраняет и восстанавливает layout зрения и M-RoPE в HIP:** `SaveSnapshot` кладёт `vision_layout_.Prefix(valid_context)`, `RestoreSnapshot` восстанавливает его ([executor.cpp:174-207](../../src/models/qwen/hip/executor.cpp#L174-L207)) через `DeviceInput::RestoreLayout` ([device_input.hip:57-120](../../src/models/qwen/vision/device_input.hip#L57-L120)), строя `DeviceRope` с `prefix_length` и `delta`; и `PrefillChunkTask` ([prefill_chunk.cpp:118-167](../../src/models/qwen/hip/prefill_chunk.cpp#L118-L167)), и `ForwardToken` ([decode.cpp:13-120](../../src/models/qwen/hip/decode.cpp#L13-L120)) используют `vision_input_`, а `replaying_ssm_state_` сбрасывается при обычных вызовах `ForwardPromptBatch` ([prefill.cpp:75-76](../../src/models/qwen/hip/prefill.cpp#L75-L76)) и `ForwardToken` ([decode.cpp:27-30](../../src/models/qwen/hip/decode.cpp#L27-L30)).
- **Идентичность изображения** хеширует пиксели/сетку/энкодер ([prompt.cpp:279-321](../../src/models/qwen/vision/prompt.cpp#L279-L321)); одноразовая токенизация системного префикса (T2a-2) не меняет токены.

**Ответ на главный вопрос Фазы 1:** это **не баг** логики/позиций/состояния при кэшировании vision-продолжения. Это расхождение жадного argmax из-за документированной неэквивалентности decode- и prefill-арифметики MoE-пути (приближённая арифметика, контракт `Q1`), проявившееся на почти равной паре кандидатов `,`/`—`.

## 4. План Фазы 2 (Вариант 2, согласованный владельцем)

1. **Mac, тестовая правка:** добавить диагностический режим в `vision_serving_test.cpp` (env `GUFO_VISION_SERVING_DIAGNOSTIC=1`, только печать без изменения проверок): сравнение по шагам зазора `top1−top2`, `maxΔ` и `KL` логитов decode-истории против холодного prefill.
2. **Стенд:** изолированная сборка цели `qwen_vision_serving_test`, короткий прогон в окне по практике прежних verify-окон с возвратом `b81812aa`.

---

## 5. Первое диагностическое окно — 2026-11-18 (частичный замер, оценка исправлена в §6)

- Изолированная сборка на `faex2` (`~/gufo-vision-diag-20261118`, патч `f3d73d69`, бинарник `49df9476…`), окно **23:01:09–23:02:08 МСК (59 с простоя)**, `LLM_FAEX2/results/2026-11-18-vision-diag/`.
- В этом прогоне подтверждены `cached_trajectory_matches_serving=1` и минимальные зазоры `top1−top2` на шаге `3`: `0.03` (cached, токен `11`) и `0.07` (cold, токен `1892`).
- **Ограничение прогона 2026-11-18 (устранено в §6):** `DumpQwenVisionContinuationDiagnostic` не вывела `max_delta` и `kl` для каждого шага отдельно — печатался только агрегат по всем 16 шагам (`19.33` / `16.73`), засорённый шагами `4..15` после расхождения контекстов. Первоначальная оценка «~0,03–0,1 логита» из разницы зазоров была **математически неверной** (разница зазоров не ограничивает `maxΔ` сверху).

---

## 6. Повторное диагностическое окно с точными построчными `maxΔ` и `KL` — 2026-11-19

**Исполнение.** В [`DumpQwenVisionContinuationDiagnostic`](../../tests/models/qwen/vision_serving_test.cpp#L318-L448) (коммит `e8c46b4c`) добавлены: построчная печать `max_delta` и `kl` на каждом шаге (в отдельном `std::ostringstream` с точностью 6 знаков), отдельные агрегаты строго по общему префиксу `0..divergence`, проверка `cold_trajectory_matches_serving` и печать логитов соперничающих токенов на шаге расхождения. Изолированная сборка на `faex2` (`~/gufo-vision-diag-20261119`, база `ac691117` + патч ≡ `e8c46b4c`, SHA бинарника `09bd137b…`); короткое окно **00:18:07–00:19:07 МСК (60 с простоя)** с `trap`-гарантией возврата `b81812aa`. После окна прод полностью верифицирован (`READY`, `child_pid=281888`, SHA `e5af3a93…`, слоты `4` idle, `"28+28="` → `'56'`, `"red"` PNG → `'red'`, журнал юнита чист). Сырые логи — `LLM_FAEX2/results/2026-11-19-vision-diag/`. `rc=1` ожидаем: тест не ослаблялся.

**Точные измеренные числа (`vision-diag-full.log`):**

- `cached_trajectory_matches_serving=1`, `cold_trajectory_matches_serving=1`, `prompt=135`, `cache_prefix=110`.
- Все оракулы точные (`append_max_logit_error=0` / `0.00` ×8, `live-history oracle exact=1` ×3, чанк-инвариантность prefill пройдена).
- **Агрегаты строго по общему префиксу (`common_prefix_steps=4`, шаги `0..3` с идентичной входной историей):**
  - `common_max_logit_delta = 0.513805` (на шаге `0`)
  - `common_max_KL_temperature1 = 0.001963` (на шаге `2`)
  - `first_divergence_step = 3`
  - *(По всем 16 шагам `all_steps_max_logit_delta = 19.332504`, `all_steps_max_KL_temperature1 = 16.729598` — доминированы разными контекстами на шагах `4..15`.)*

| Шаг | `cached_token` | `cold_token` | `cached_margin` | `cold_margin` | `max_delta` | `kl` (T=1) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 40 | 0.918016 | 0.809122 | 0.513805 | 0.001420 |
| 1 | 5312 | 5312 | 2.266123 | 2.192135 | 0.332067 | 0.000269 |
| 2 | 2438 | 2438 | 0.733339 | 0.678083 | 0.312154 | 0.001963 |
| **3** | **11** | **1892** | **0.030472** | **0.070221** | **0.391665** | **0.001826** |

- **Логиты соперничающих токенов на шаге расхождения (`step=3`):**
  - `token_a = 11` (`,`): `L_cached = 22.374062`, `L_cold = 22.281139` (`Δ = +0.092923`)
  - `token_b = 1892` (`—`): `L_cached = 22.343590`, `L_cold = 22.351360` (`Δ = -0.007770`)

**Итоговый вывод (Вариант 3 подтверждён точными измерениями):**
Реальный `max_delta` между decode-историей (`ForwardToken`) и матричным prefill (`ForwardPromptBatch`) на общих шагах `0..3` составляет **`0.31–0.51` логита** (что укладывается в известный по [INVESTIGATION_RESULT.md](INVESTIGATION_RESULT.md) §В диапазон `0.09–0.82`), а вероятностные распределения близки (`KL <= 0.001963`). На шаге `3` из-за экстремально узкого зазора (`0.030472` / `0.070221`) сдвиг логита токена `11` на `+0.092923` переворачивает жадный `argmax`. Дефектов позиций/состояния/снимка нет. По решению владельца проверка не ослабляется (`35/36`), а вопрос приёмки закрывается вместе с контрактом `Q1`.
