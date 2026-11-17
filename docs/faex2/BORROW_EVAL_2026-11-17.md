# BORROW_EVAL_2026-11-17: Оценка заимствований по пунктам ROADMAP_SPEC_01.md

Дата: 2026-11-17 (обновлено по итогам двойного аудита)  
Статус: **уточнённый отчёт для решения владельца; не разрешение на реализацию**  
Ревизия анализа: **`4edd4d0b`** / база `a9e29ce1` ветки `faex2` (рабочий бинарник baseline `004d8fa1`; дорожная карта — `90f73f1a` с правкой `26904442`); апстрим `upstream/main` = `12e8b99f` (2026-11-09, #934).

---

## 0. Назначение, метод и границы достоверности

Документ отвечает на три вопроса по каждому пункту заимствований из [ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md): **нужно ли** (закрывает ли реальную проблему проекта), **целесообразно ли** (соотношение эффекта, сложности и риска с учётом уже сделанного) и **возможно ли** (техническая переносимость + правовая чистота).

**Метод:** только чтение, сверка кода и git-истории (включая проверку апстримных патчей на `HEAD`, проверку границ ядер/буферов и пересчёт байтовых бюджетов); движок не менялся, ничего не собиралось, стенд FEVM FAEX2 (`192.168.3.80`) в момент проверки не имел доступного SSH на порту 22 (ICMP `ping` ~10,8 мс, TCP/22 `Operation timed out`), поэтому живые замеры не запускались. Все внешние цифры взяты из документов самих источников и помечены как их замеры, требующие воспроизведения на нашем стенде (см. [BORROW_PLAN_2026-11-17.md](BORROW_PLAN_2026-11-17.md)).

**Источники и уточнённые факты о них:**

| Проект | Где | Что это на самом деле | Проверенная база / доказательства |
| :--- | :--- | :--- | :--- |
| **`gufo-laguna`** | В нашем дереве: [src/models/laguna/](../../src/models/laguna/) | HIP-движок с пулом графов (C1) и NPU | [src/models/laguna/gpu/engine.cpp](../../src/models/laguna/gpu/engine.cpp) |
| **апстрим `gufo`** | remote `upstream` (`12e8b99f`) | Исходная линия (`#934`..`#1122`) | `git show <hash>`, `git merge-base --is-ancestor` |
| **`tubo`** | `_borrow-eval/tubo` | gufo-форк (MIT): **Qwen3.8-27B** (dense), `turbo4/5/6` KV, `GUFO_DEV_MEMTRIM_*`, `snapshot_copy.hip` | merge-base `c92b19c5` (#850, 2026-10-27), 24 коммита (`c92b19c5..HEAD`), 341 файл, `+93 591 / -2 038` |
| **`NinjaPear/Q6dense`** | `_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense` | gufo-форк (MIT): полноценный С++/HIP-бэкенд под **Qwen3.6-35B-A3B** + рецепт GGUF | HEAD `01426f6d` (`Merge: d7e938e4 159f2b84`), база upstream `159f2b84` (`#984`, Nov 3; `0.5.0` — `23cacbb9`), 3 коммита форка, 134 файла, **`+53 350 / -60`** |
| **`strixite`** | `_borrow-eval/strixite` | Независимый HIP-движок с нуля (**AGPL-3.0**, тег `v0.7.0` `0945ce8`) для Qwen3.8-Flash-Next — **только идеи и архитектура, ни одной строки кода!** | [LICENSE](../../../_borrow-eval/strixite/LICENSE), [THIRD_PARTY.md](../../../_borrow-eval/strixite/THIRD_PARTY.md); см. отдельный **§9** |

---

## 1. Сводная таблица вердиктов (согласованная с ROADMAP_SPEC_01)

| № | Пункт ROADMAP_SPEC_01 | Источник | Необходимость | Целесообразность | Возможность | Вердикт |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | §1.2 п.1, §3.4 Бакетные HIP-графы + device-pos split-K + прогрев | `gufo-laguna`, `Q6dense` (MIT); *(в `strixite` графов 0)* | **Высокая** (горячий путь MTP/DFlash — `ForwardVerificationBatch` и `MTP` вообще без графов; паузы C4 27–39 мс и 1-го запроса 315 мс) | **Высокая** (Этап 2; критично разделить `C1` и `C4`) | Да, ~**600–1200 строк** (вкл. `wave64`, verify, MTP, C4 и прогрев `hipBLASLt`) | **ДЕЛАТЬ (Этап 2)** по уточнённому ТЗ |
| 2 | §1.2 п.2, §2.Б п.7 MoE-роутинг, grouped/indirect GEMM и просадка `MTP2/3` | `laguna`, `Q6dense` (MIT), `strixite` (идея); *(`tubo` 27B — без MoE)* | **Не нужна** переписка с нуля (`gate/up` grouped/indirect уже у нас), но **нужна диагностика** `MTP2/3` (−5..−33 %) и `down_exps` | **Низкая** для стороннего роутинга; **средняя** для grouped `down_exps` (`K=512`, `Q4/Q5/Q6`) из `Q6dense`, если профиль докажет | Да (`by_expert` сейчас откл. при `K!=640`) | **НЕ ЗАИМСТВОВАТЬ** роутинг; **не закрывать** §1.2 п.2 до профиля `MTP2/3` и проверки grouped `down_exps` |
| 3 | §1.2 п.3, §3.5 Cost gate спекуляции | Апстрим `1167a944`, `Q6dense` (`0.5`), `strixite` (`capture`/`MtpPick`); *(`laguna` — нет гейта)* | **Средняя/Условная** (замена жёсткого `GUFO_MTP_MOE_MAX_COHORT=4` нужна и без `DFlash2`) | Высокая для `1167a944` (но для `27B`, не 35B); высокая для 35B-гейта после Этапа 0 | Да (~20–30 строк в 27B; ~150–250 в 35B) | **`1167a944`** — как мелкий долг 27B; **35B cost gate — ДЕЛАТЬ ПОСЛЕ Этапа 0** |
| 4 | §1.2 п.4, §3.2 п.2 Вариант А: `GUFO_QWEN_SPARSE_ATTN=1` | Собственный код `34e61a15`, уже в дереве и уже в **Этапе 2 п.1** | Зависит от телеметрии длин Hermes | **Очень высокая** (0 строк кода, дешёвый тест; ~+26 % в рабочем `MTP1`) | Да (уже в дереве) | **ДЕЛАТЬ ПЕРВЫМ на Этапе 2** (после телеметрии Этапа 1, **ДО** `Q8_0`) |
| 5 | §1.2 п.4, §3.2 п.2 Вариант Б: KV `Q8_0` (+ `kTurbo6`) | `Q6dense` (MIT: 8-бит KV под 35B), `tubo` (MIT: уроки), `strixite` (`kv8` идея) | **Условная** (при 4×128K память не дефицит; нужно только при частых >32K **и** провале `sparse`) | Средняя для `Q8_0` (−4,69 ГиБ / −5,03 ГБ); низкая для `kTurbo6` (compute-bound на verify в `tubo` N32) | Да (`Q8_0`: 300–600+ строк; `turbo6`: ~1000+) | **`Q8_0` — УСЛОВНО (Этап 3)** только после отказа `sparse`; **`kTurbo6` — далёкий резерв** |
| 6 | §1.2 п.5, §3.1, Этап 4 `Q6dense` (`Q6_K` dense + `UD-Q4_K_XL` experts) и `.ghalo` | `Q6dense` (рецепт + MIT-код `qwen36_35b_a3b`), `strixite` (идеи `wmma`/`strixw`) | Низкая для качества (паритет 7,25/7,62), средняя/высокая для скорости 64K–131K | **Высокая** для дешёвого GGUF-теста (0 строк); **средняя** для точечного MIT-порта `DenseQ6KVec`/`MtpHiddenKernel` | Да (проверить лицензию HF и `q6dense.types`) | **ПОПРОБОВАТЬ** GGUF-рецепт + изучить MIT-оптимизации `Q6dense`; **`.ghalo` — ОТЛОЖИТЬ** |
| 7 | §1.2 п.6, §2.В п.11, Этап 1 п.1–2,4 Hermes Agent: API, телеметрия, планировщик, кэш | Апстрим `6d793814`, `4ab04ab8`, `fe2694bc`, `8ab834a5`; `strixite` (только идеи, см. **§9**) | **Высокая** для всех 4 частей (55K-префикс, блокировка семьи, `aux_title_request.json`, сжатие) | **Высокая** (`/tokenize` + `input_tokens` вручную; `ram_only` / `System+Checkpoint` / `resume_loss` / `capture` концептуально; приоритеты в планировщике) | Да (API 240–330 строк вручную; идеи `strixite` — с нуля) | **ДЕЛАТЬ на Этапе 1** (в полном составе всех 4 подзадач §1.2 п.6) |
| 8 | §2.В п.9–10, §3.2 п.3, Этап 3 п.1 Потоковый `Restore` и зазор памяти +4..7,5 ГиБ | `tubo` (MIT: `GUFO_DEV_MEMTRIM_9`, `snapshot_copy.hip`), `strixite` (идеи `MemoryGovernor`, `HostBuffer`, `plan_memory`) | **Средняя/Высокая** (сейчас `Restore` читает весь файл в RAM; все 10 ГиБ KV выделяются сразу) | **Высокая** (`MEMTRIM_9` прямо бьёт по +4..7,5 ГиБ к `llama.cpp`; `snapshot_copy.hip` ускоряет чанки `Restore`) | Да (MIT-код `tubo` + чистые идеи) | **ДОБАВИТЬ в Этап 3** (идеи `MemoryGovernor`/`HostBuffer` полезны и на Этапе 1/2) |
| 9 | §1.3, §2.А (`35/36`, `ce617610`), §3.3 NPU (`649824af`) | Апстрим, `Q6dense` (#16 диаг. PV), `tubo` (`NORMQ`) | Подтверждено без изменений | `ce617610` конфликтует с fused-attention; `1b1dfa80` уже отклонён; диаг. PV у нас уже есть | — | **Не трогать раньше срока** (§2.А `35/36` — своя отладка; `649824af` — в Этап 5) |

**Правовое правило номер один (см. §10):** [strixite](../../../_borrow-eval/strixite/LICENSE) — **GNU AGPL-3.0**. Из него **нельзя переносить ни одной строки кода** в наш движок (наш `gufo serve` — сетевой сервис, AGPL §13 потребовал бы раскрыть исходники всего форка). Заимствовать можно **только архитектурные идеи**, реализуя их с нуля под наш код. Подробная карта всех таких идей — в **§9**. Из `tubo` и `Q6dense` (MIT) код можно переносить с сохранением copyright-нотисов.

---

## 2. Пункт 1 — Бакетные HIP-графы, device-pos split-K и прогрев (roadmap §1.2 п.1, §3.4)

### 2.1. Что у источников (`gufo-laguna`, `Q6dense`; в `strixite` графов нет)
* **`strixite` не подходит:** несмотря на упоминание в таблице roadmap §1.2 п.1, в `strixite` вообще **нет** `hipGraph` / `hipStreamBeginCapture` (`0` графов).
* **`gufo-laguna` ([engine.cpp](../../src/models/laguna/gpu/engine.cpp)):**
  * пул `StepGraph graphs[Session::kMaxVerifyRows + 1]` на 17 слотов ([engine.cpp:1391](../../src/models/laguna/gpu/engine.cpp#L1391)), захват на 2-м шаге ширины (`g.steps > 1`), флаг `g.failed` без повторных попыток ([engine.cpp:1401-1420](../../src/models/laguna/gpu/engine.cpp#L1401-L1420));
  * `pos` передаётся через `hipMemcpyAsync` из pinned host-буфера в `d_pos` внутри графа;
  * захват разрешён только при `position + n >= kSplitAttnMinDepth` ([engine.cpp:1396-1397](../../src/models/laguna/gpu/engine.cpp#L1396-L1397)), поэтому внутри графа навсегда фиксируется именно split-K ветка.
* **`Q6dense` (MIT, [executor.cpp](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/executor.cpp)):**
  * уже реализовал пул HIP-графов **специально для Qwen3.6-35B-A3B**: отдельные ключи для decode, verify, `ForwardBatch`, `MtpForward`, `MtpDraftChain`, с device-указателем `*start_pos` в ядре внимания;
  * **важнейший урок (`TrimRollback`, [executor.cpp:192-206](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/executor.cpp#L192-L206)):** verifier-графы запекают в себе указатели буферов отката (`rollback` / recurrent state); при уменьшении глубины отката или перевыделении буферов эти графы **обязательно нужно уничтожать**, иначе replay ударит по освобождённой памяти!

### 2.2. Реальное состояние `gufo-faex2` (что уточнено при аудите)
1. **Где сейчас графы есть, а где идёт реальная работа при MTP/DFlash2:**
   * [`HipGraphDecodeExecutor`](../../src/core/hip/detail/hip_graph_decode_executor.hpp) держит один `instance_` и применяется **только** в [`ForwardToken`](../../src/models/qwen/hip/decode.cpp#L50-L108) (одиночный AR-декод `T=1`). В `decode.cpp:63-66` аргумент `moe_enabled` даже не передаётся (остаётся `false`, `kMoeNotCapturable` не выставляется), а `WarmMoeDecode` уже вызывается перед захватом (`decode.cpp:85-88`).
   * При рабочем **`MTP1` или `DFlash2` `ForwardToken` вообще не исполняется**: целевая верификация идёт через [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564) (`T=2..8` и более), многопользовательский AR — через [`ForwardTokenBatch`](../../src/models/qwen/hip/batched_decode.cpp#L236), а шаг черновика MTP — через [`QwenMtpGpuExecutor::Run`](../../src/models/qwen/hip/mtp/executor.cpp#L232). Во всех трёх путях HIP-графов сейчас **нет вовсе**.
2. **Какое split-K ядро реально работает на 35B-A3B:**
   * По [`LaunchSplitKAttentionHalfPartials`](../../src/models/qwen/hip/kernels/attention_decode.hip#L923-L956) на хосте проверяется `const bool deep = static_cast<std::size_t>(start_pos) + 1 >= 512;`.
   * Для 35B-A3B (`num_heads == 8 * num_kv_heads && paired_kv`): при `128 <= seq_len < 512` запускается `QwenDecodeSplitKAttentionHalfPartialsKernel<>`, а при **`seq_len >= 512`** запускается **`QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>`** ([attention_decode_verify_wave64.hip:24-29](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip#L24-L29)).
   * **Оба** ядра принимают `pos` скаляром (запекается в граф!). Если забыть `attention_decode_verify_wave64.hip` и порог `512` (например, захватить граф при `pos = 200`), то на `64K–128K` граф навсегда запечёт более медленное 32-lane ядро вместо wave64 `6bcf0c3b`!
   * В [`QwenDecodeSplitKAttentionReduceKernel`](../../src/models/qwen/hip/kernels/attention_decode.hip#L836-L839) **нет параметра `pos`**, а `split_count` при split-K всегда равен `32` — менять его сигнатуру не нужно. [`SplitKDecodePartition`](../../src/models/qwen/hip/kernels/attention_decode_split_k.hpp#L49-L56) **уже `__device__ __forceinline__`**.
3. **`d_decode_step_params` и разница `C1` vs `C4`:**
   * Буфер `d_decode_step_params` действительно нужно создать (образец `d_pos` — [`WriteKVCachePtrKernel`](../../src/models/qwen/hip/kernels/attention_decode_graph.hip#L14-L32)).
   * Простой паттерн `laguna` (`pos + row`) корректен **только для одной сессии (`C1`)**. Для многопользовательских пауз `C4` (`27–39 мс`, `ForwardVerificationBatch` с `items.size() > 1`) у каждой сессии своя позиция `pos[i]` и свои указатели арены/KV — без отдельного device-массива параметров по слотам графы ускорят только `C1`.
4. **Прогрев на старте (§3.4 п.3):** в roadmap входит принудительный прогрев `hipBLASLt` и `WarmMoeDecode` до приёма 1-го HTTP-запроса (убирает ленивую загрузку ELF 3,6 МиБ и паузу 265/315 мс).

### 2.3. Необходимость, целесообразность, объём и вердикт
Необходимость и целесообразность **высокие** (без графов на verify + MTP + прогрева мы теряем скорость `MTP1` и ловим всплески 27–39 / 315 мс). С учётом `attention_decode_verify_wave64.hip`, `ForwardVerificationBatch`, `QwenMtpGpuExecutor`, разделения `C1/C4`, инвалидации при `TrimRollback` и стартового прогрева реальный объём — **~600–1200 строк** (вместо прежней заниженной оценки 390–700).

**Вердикт: ДЕЛАТЬ в рамках Этапа 2**, переписав ТЗ `T3–T5` с учётом `wave64`, `MTP`, `C1/C4`, `TrimRollback` (`Q6dense`) и прогрева `hipBLASLt`.

---

## 3. Пункт 2 — MoE-роутинг, grouped/indirect MoE GEMM и просадка `MTP2/3` (roadmap §1.2 п.2, §2.Б п.7)

### 3.1. Что уже есть у нас, а где обнаружен реальный пробел
* **Indirect и grouped для `gate/up` и крупных батчей уже есть:** [`qfn_mmq_moe_gated_vec`](../../src/models/qwen/hip/ops/moe_vec.hpp) → `mul_mat_vec_q_moe`, `group_moe_slots` (`mmvq.hip.cpp:563`), `launch_moe_grouped` (`:805`), `group_moe_batch` (`:757, 1437`), `MMVQ_MAX_ROUTED_BATCH = 64`; prefill идёт через [`QwenMoePrefill`](../../src/models/qwen/hip/moe_prefill.hpp) (`RoutedCompactRows` + `RoutedF16Gemm`). «Все 256 экспертов» на decode/verify нигде не читаются. (Примечание к roadmap §1.1: вместо исторического имени `GetMoeFfnScratch` сейчас используется `QwenMoeScratch` в [`executor.hpp:320-333`](../../src/models/qwen/hip/executor.hpp#L320-L333).)
* **Важный нюанс по `down_exps` в `35B-A3B`:** в [`batched_decode.cpp:133-136`](../../src/models/qwen/hip/batched_decode.cpp#L133-L136) down-проекция вызывается как `qfn_mmq_moe_vec(..., K=ff, n_tokens=n*used, n_expert_used=1, ...)`. А в [`qfn_mmq.hip.cpp:505-508`](../../src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp#L505-L508) группировка `by_expert` включается только при условии:
  `!gated && !W_b && !X_q8 && (type == Q5_1 || type == Q8_0) && K == 640 && n_expert_used == 1 && n_tokens > 1`.
  Поскольку у **Qwen3.6-35B-A3B** `expert_ff = 512` (не `640`) и кванты `UD-Q4_K_XL` (`Q4_K/Q5_K/Q6_K`), `by_expert` на нашей модели всегда `false`, то есть слоты экспертов в `down_exps` не группируются!

### 3.2. Исправление ошибок анализа других проектов
* **Опровержение ошибки про `strixite::mtp_pick`:** ядро [`strixite/kernels/mtp_pick`](../../../_borrow-eval/strixite/kernels/mtp_pick.hpp) — это **НЕ MoE-роутинг** (и не имеет отношения к `LaunchMoeRouterTopK`). Оно выбирает **argmax и 2-е значение по логитам словаря черновика MTP** (+ NaN и 9 слов ошибок GPU за один 64-байтный copy). У нас эквивалент экономии копий словаря уже реализован через [`LaunchBatchedGPUArgmax`](../../src/models/qwen/hip/mtp/executor.cpp#L342), а идея читать ещё и маржу top-2 относится к **cost gate** (§4 и §9).
* **MIT-решение в `Q6dense` (`EXPERIMENTS.md` #21 vs #25):** сам `Q6dense` отклонил *in-kernel* grouping (#25 — «slower»), но **оставил** группировку строк `expert-down` для `Q4_K/Q5_K/Q6_K` до 64 строк в общем MMVQ (#21 — побитово идентично для каждой строки).
* **Про `tubo dflash`:** отсутствие упоминаний MoE в `tubo` объясняется тем, что `tubo` сделан для плотной модели Qwen3.8-27B.

### 3.3. Вердикт
**НЕ ЗАИМСТВОВАТЬ** сторонние архитектуры MoE-роутинга из `laguna`/`strixite`/`tubo`. При этом **не закрывать** §1.2 п.2 в roadmap как «полностью решено», пока профилирование не объяснит просадку `MTP2/3` на 2–4 потоках (−5..−33 %, §2.Б п.7 — включая MTP-внимание и отсутствие графов) и не покажет, оправдана ли доработка `by_expert` в `qfn_mmq.hip.cpp` для `K = 512` (`Q4_K/Q5_K/Q6_K`) по образцу MIT `Q6dense`.

---

## 4. Пункт 3 — Cost gate спекуляции и управление черновиком (roadmap §1.2 п.3, §3.5)

### 4.1. Источники
* **`gufo-laguna`:** проверено в [src/models/laguna/](../../src/models/laguna/) — cost gate и адаптивная политика ширины отсутствуют (брать нечего).
* **Апстрим `1167a944` (#703, `qwen38_flash_next`) и `2caf2a82` (#664):** `2caf2a82` уже отражён у нас через cohort; `1167a944` (+35/−13, 4 файла) добавляет чистую функцию [`MtpPolicyConcurrency`](../../src/models/qwen38_flash_next/mtp_policy.hpp) (различает seeded и unseeded sampled; `SamplerConfig::seed` уже есть в [`sampling.hpp:44`](../../src/core/sampling.hpp#L44)). **Важно:** этот патч касается ветки `qwen38_flash_next` (27B), а не целевой `src/models/qwen/` (35B-A3B).
* **`Q6dense` (MIT, `EXPERIMENTS.md` #10):** `VerifyGreedyDraft` (`kMinDraftProbability = 0.5F`) — прекращает верификацию хвоста черновика, если уверенность draft-токена ниже `0.5` (~+1 % скорости без изменения greedy-вывода).
* **`strixite` (AGPL — только идеи):** лог `.cap` ([`capture.hpp`](../../../_borrow-eval/strixite/serve/capture.hpp)) для офлайн-прогона порогов политики без GPU и маржа логитов `best_v - second_v` ([`mtp_pick.hpp`](../../../_borrow-eval/strixite/kernels/mtp_pick.hpp)).

### 4.2. Что у нас в 35B-A3B и почему cost gate нужен при любом исходе Этапа 0
В [`src/models/qwen/hip/auto_draft.cpp`](../../src/models/qwen/hip/auto_draft.cpp#L183-L263) выбор идёт по `tokens_per_second`, при `temperature > 0` принудительно ставится DFlash, а в [`mtp/backend.cpp:265-292`](../../src/models/qwen/hip/mtp/backend.cpp#L265-L292) и [`mtp_policy.hpp:85-90`](../../src/models/qwen/mtp_policy.hpp#L85-L90) действует жёсткий статический порог `GUFO_MTP_MOE_MAX_COHORT = 4`. Таблицы стоимости для DFlash уже есть в [`dflash_policy.hpp:321-360`](../../src/models/qwen/dflash_policy.hpp#L321-L360).

Даже если `DFlash2` провалит порог `15 %` на Этапе 0 и будет отклонён, **35B-A3B всё равно нужен cost gate** (~150–250 строк) для динамического решения **`MTP1 ↔ AR` на C2–C8** с учётом реального acceptance, числа активных слотов, длины контекста и порога уверенности черновика (вместо жёсткой отсечки по `4` потокам). При прохождении `DFlash2` туда же добавляется переключение `MTP1 (</think>) → DFlash2 (JSON/tools)`.

**Вердикт:** `1167a944` (`T6`) — можно забрать как мелкую безопасную уборку долга в `27B` (низкий приоритет); 35B cost gate (`T7`) — **ДЕЛАТЬ ПОСЛЕ Этапа 0** с учётом идей `0.5`-фильтра (`Q6dense`) и офлайн-симуляции (идея `capture` из `strixite`). Апстримные `kMtpCycleMilliseconds` не копировать — нужна своя калибровка.

---

## 5. Пункт 4 — Длинный контекст: готовый `GUFO_QWEN_SPARSE_ATTN=1` перед условным `Q8_0` KV (roadmap §1.2 п.4, §3.2 п.2)

### 5.1. Исправление критического перекоса приоритетов
По ревизии `90f73f1a` в [ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md#L39) **строго задан порядок**:
1. В **Этапе 1** — телеметрия реальных длин историй Hermes Agent (пока неизвестно, какая доля диалогов превышает 32K–64K).
2. В **Этапе 2 п.1** — дешёвая проверка **уже готового** [`GUFO_QWEN_SPARSE_ATTN=1`](../../src/models/qwen/hip/kernels/attention_sparse.hip) (`34e61a15`, `0` строк нового кода). При дефолтах прореживание начинается с `290` блоков (`18 560` токенов), формальный порог включения пути `sparse_min_context = 8192` ([`numerical_overrides.hpp:42`](../../src/models/qwen/hip/numerical_overrides.hpp)). Важное уточнение масштаба эффекта: исторические **+77 %** (`24,4 → 43,2 tok/s`) были на связке `AR + NPU`, а для нашего рабочего **`MTP1`** ускорение составляло **`25,7 → 32,4 tok/s` (~+26 %)**.
3. Только **УСЛОВНО в Этапе 3** — тяжёлая разработка `Q8_0` KV (~2–3 недели работы), и **только если** телеметрия подтвердит частые длинные истории, **а** `sparse` не устроит по качеству (при этом `sparse + Q8_0` вместе строго запрещены!).

### 5.2. Точная арифметика KV-кэша (устранение путаницы `ГиБ` / `GB`)
Для Qwen3.6-35B-A3B (`10` слоёв Full Attention × `2` KV-головы × `head_dim = 256` = `512` элементов на плоскость; K+V = `10 240` элементов/токен), профиль `4 × 131 072 = 524 288` токенов:

| Формат | Байт / значение | Байт / токен | Объём на `4 × 128K` | Экономия к `FP16` | Источник |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FP16`** | `2,000` (16 бит) | `20 480` Б | **10,00 ГиБ** (**10,74 ГБ**) | — | текущий `kFp16` ([execution_policy.hpp](../../src/models/qwen/hip/execution_policy.hpp)) |
| **`Q8_0`** | `34 / 32 = 1,0625` (8,5 бит) | `10 880` Б | **5,31 ГиБ** (**5,70 ГБ**) | **4,69 ГиБ** (**5,03 ГБ**) | GGML `Q8_0`; ядро чтения под 35B есть в `Q6dense` |
| `turbo4` | `66 / 128 = 0,515625` | `5 280` Б | 2,58 ГиБ (2,77 ГБ) | 7,42 ГиБ (7,97 ГБ) | `tubo` (`turbo4_codec.hpp:75-93`) |
| `turbo6` | `98 / 128 = 0,765625` (6,125 бит) | `7 840` Б | **3,83 ГиБ** (**4,11 ГБ**) | **6,17 ГиБ** (**6,63 ГБ**) | `tubo` (`turbo4_codec.hpp:75-93`) |

### 5.3. Что брать у `Q6dense`, `tubo` и `strixite` для условного `Q8_0` (и чего НЕ брать)
* **Важная поправка по `tubo`:** у `tubo` **НЕТ `kQ8_0`** ([`execution_policy.hpp:15-37`](../../../_borrow-eval/tubo/src/models/qwen/hip/execution_policy.hpp#L15-L37) содержит только `kFp32, kFp16, kTurbo4, kTurbo5, kTurbo6` для 27B). Его ценность для нас — **три архитектурных урока** (а не формат `Q8_0` и не 4256-строчное ядро `turbo4_attention.hip`):
  1. **Enum дописывать только в конец** (`kFp32=0, kFp16=1, ...`) и сразу поднимать precision-bound в [`arena.cpp:252-256`](../../src/models/qwen/hip/arena.cpp#L252-L256), чтобы не ломать чтение старых снимков и собственных новых.
  2. **Таблица форматов по плоскостям (`QwenKvLayerFormat`)**, если позже понадобится гибридный KV (разные слои с разной точностью).
  3. **Единая функция расчёта смещения блока** (`QwenKvHeadBlockOffset` в `kv_read.hpp`) для writer и reader + обязательный **CPU-оракул кодека**.
* **Что есть у `Q6dense` (MIT) именно для 35B-A3B:** готовые [`QuantizeKvKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L1704-L1723) и [`GqaDecodeAttentionWmmaKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L1744-L1825) с 16-байтными загрузками, `*start_pos` из device-памяти и FP64-тестами под геометрию `16Q/2KV, head_dim=256`. *Ограничение:* `Q6dense` хранит 8-битный кэш *дубликатом* рядом с `F16` (без экономии VRAM); нам же для экономии **4,69 ГиБ** нужна замена `FP16` (через доработку нашего wave64 split-K пути либо адаптацию ядра `Q6dense` под однослойное хранение + префилл-чтение).
* **Опасная ошибка старого плана:** [`DequantizeQ8_0ToBf16Kernel`](../../src/models/qwen/hip/kernels/prefill_gemm.hip#L382-L408) деквантует веса линейных матриц в **`BF16`**, а наше внимание (`attention_wmma.hip`, `attention_decode_verify_wave64.hip`) работает в **`FP16`** — это ядро напрямую непригодно!
* **Идеи `strixite::kv8` (см. §9.4):** поканальная шкала для `K` (−42 % ошибки на K после RoPE), точный несжатый хвост `< 32` строк (`kv8_exact_from`) и предупреждение, что распаковка 8-бит внутри WMMA-цикла prefill тормозит ядро в **2,7×**.
* **Почему `kTurbo6` — далёкий и сомнительный резерв:** помимо >1000 строк нового кодека/ядер, в замере `tubo` `BENCHMARKS.md` (`N32`) 8-строчное verify-чтение `turbo6` упёрлось в инструкции/LDS (**68,4 ГБ/с**, в сервере **49 ГБ/с**), что крайне невыгодно для нашего `MTP1` (многострочный verify) при достаточных 128 ГБ памяти.

**Вердикт:** **Этап 2 → `GUFO_QWEN_SPARSE_ATTN=1` (`T15`, первым)**; **`Q8_0` KV (`T8–T12`) → УСЛОВНО на Этапе 3** (с уроками `tubo`, базой `Q6dense` и идеями `kv8` `strixite`); `kTurbo6` (`T13`) → далёкий резерв.

---

## 6. Пункт 5 — Эксперимент `Q6dense` (`Q6_K` dense + `UD-Q4_K_XL` experts) и `.ghalo` (roadmap §1.2 п.5, §3.1, Этап 4)

### 6.1. Что представляет собой `Q6dense` на самом деле
Это и **рецепт смешанного GGUF-квантования** (из `unsloth BF16` `5bc3e238` + `imatrix` + `q6dense.types`, в клоне отсутствуют), и **специализированный MIT-движок** [`src/models/qwen36_35b_a3b/`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/) (+53 350 строк от апстрима `159f2b84`), показавший на `gfx1151` 128 ГБ с их ядрами **101,3 tok/s** MTP на 8K и **73,6 / 66,85 tok/s** MTP на 64K / 131K (`Q6dense BENCHMARKS.md`), при судейском паритете против `UD-Q4_K_XL` (7,25 vs 7,25; 7,62 vs 7,62) и готовых отрицательных проверках (`Q5_K` dense и `Q4_K_M` dense отклонены).

### 6.2. Совместимость рецепта и ценность MIT-оптимизаций
* **GGUF-рецепт (`T14`):** в [`weights.cpp:62-95`](../../src/models/qwen/weights.cpp#L62-L95) загрузчик принимает все K-кванты и `Q8_0` для `kEmbedding`/`kProjection`, но требует строго **`F32` для `kNorm` и `kSsmParameter`** и `SupportsMoeRouterType` для роутера. Поэтому **обязательный шаг перед скачиванием BF16 — проверить лицензию карточек HF и типы в `q6dense.types`**. Эксперимент стоит 1 вечер стенда и **0 строк кода движка**.
* **Что в MIT-коде `Q6dense` реально дало скорость (и стоит изучить в `T14b` после замера GGUF):** перепаковка `Q6_K` dense при загрузке и [`DenseQ6KVecKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L262) (`EXPERIMENTS.md` #20), grouped `expert-down` для `Q4_K/Q5_K/Q6_K` (#21), разделение `output_norm` / `nextn.shared_head_norm` в [`MtpHiddenKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L2176-L2195) (#9 — стоит проверить нашу связку в [`mtp/executor.cpp:259-261,320-322`](../../src/models/qwen/hip/mtp/executor.cpp#L259-L261)!), `SmallGemmSplitF16Kernel` для роутера (#17). Дубликаты `PrefillQ8` ([`device_model.cpp:177-190`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/device_model.cpp#L177-L190)) копировать не нужно (расходуют лишнюю память).
* **`.ghalo` (§3.1):** отложить. Опыт `Q6dense` подтверждает, что ускорение даёт перепаковка тензоров *при загрузке в память*, а не собственный дисковый контейнер. Выравнивание `TensorEntry` в `ROADMAP_SPEC_01` уже поправлено в `26904442`.

**Вердикт:** **ПОПРОБОВАТЬ** GGUF-рецепт (`T14`) в Этап 4 + при необходимости точечно заимствовать MIT-ядра из `Q6dense` (`T14b`); **`.ghalo` — ОТЛОЖИТЬ**.

---

## 7. Пункт 6 — Профиль Hermes Agent: API, телеметрия, планировщик и кэш (roadmap §1.2 п.6, §2.В п.11, Этап 1)

В `ROADMAP_SPEC_01` §1.2 п.6 и Этапе 1 п.1–2,4 поставлены **четыре** задачи под основной клиент Hermes Agent (постоянный префикс ~55 тыс. символов с 20 инструментами + служебный `aux_title_request.json` + автосжатие истории):

1. **2-я очередь Пакета B (`6d793814` `/tokenize` и `4ab04ab8` `input_tokens`) — ДЕЛАТЬ вручную (~240–330 строк):**
   * 3-way патчи конфликтуют из-за расхождения `serve` после `d0f7c3a0` (#548). Все опорные точки на месте: `register_routes` ([http_server.cpp:1851-1870](../../src/cli/serve/http_server.cpp)), `AnthropicCountTokens` (`:1328`), `OpenAiResponses` (`:1127`), `runner().Tokenize(...)` ([inference_backend.cpp:5858](../../src/cli/serve/inference_backend.cpp)), `count_chat_tokens` ([text_generation_backend.hpp:383-386](../../src/cli/serve/text_generation_backend.hpp#L383-L386)), `Tokenize` ([text_model_runner.hpp:345-348](../../src/cli/serve/text_model_runner.hpp#L345-L348)).
2. **Телеметрия реальных длин историй Hermes (Этап 1 п.1) — ДОБАВИТЬ В ПЛАН:**
   * Нет готового патча в `tubo`/`Q6dense`; **лучший образец архитектуры — `capture.hpp` из `strixite`** (идея, не код! см. §9.3).
3. **Приоритет коротких семейных запросов перед длинным агентским prefill (`text_generation_scheduler.cpp`) — СОБСТВЕННАЯ РАЗРАБОТКА:**
   * Готовой реализации ни в одном из трёх форков нет; в дополнение к уступкам внутри уже начатого prefill (Пакет D `SetPrefillYield`) нужно добавить приоритет в очереди планировщика [`text_generation_scheduler.cpp`](../../src/cli/serve/text_generation_scheduler.cpp) (плюс идея `others_decoding` из `strixite` для уменьшения чанков prefill, если декодеры ждут).
4. **Защита кэша от вытеснения `aux_title_request.json` и при сжатии истории Hermes:**
   * Проверка апстрим-коммитов в `gufo-faex2`: `68e8475d` (#362 `retain prefixes across conversation history edits`) и `93af45d4` (#348) **уже есть в `HEAD`**; а вот **`fe2694bc`** (#1104 `let optional copies displace conversations unused for one record turnover`), **`8ab834a5`** (#903) и **`b911c150`** (#964) **отсутствуют (`NOT_IN_HEAD`)** — проверить их применимость к `ContinuationCache`.
   * **Сильнейшее архитектурное решение — идеи из `strixite::prompt_cache` (`ram_only = true` для one-shot `aux_title_request.json`, `Kind::System / Checkpoint / Turn`, idle I/O + квота записи) и `strixite::resume_loss`** (см. §9.1).

---

## 8. Дополнительно по roadmap: Sparse-attention, упущенные находки `tubo`, зрение и NPU

* **Sparse-attention (`GUFO_QWEN_SPARSE_ATTN=1`):** код [`attention_sparse.hip`](../../src/models/qwen/hip/kernels/attention_sparse.hip) (`34e61a15`) **уже в дереве** и **уже запланирован на Этап 2 п.1** (`90f73f1a`). Нужно провести A1-проверку семантики MTP/rollback/границ и A2-замер качества (на `MTP1` ориентир ~+26 %, а не +55 %!).
* **Крупный пропуск по `tubo` (MIT) для техдолга §2.В п.9–10 (`Restore` и `+4..7,5 ГиБ` к `llama.cpp`):**
  * **`GUFO_DEV_MEMTRIM_9`** (динамические 2-МиБ KV-клетки) — объясняет и помогает закрыть часть зазора по памяти (мы сейчас аллоцируем все 10 ГиБ KV для 4 сессий сразу в конструкторе [`arena.cpp`](../../src/models/qwen/hip/arena.cpp)), а `MEMTRIM_4/7` убирают простаивающие буферы.
  * **[`snapshot_copy.hip`](../../../_borrow-eval/tubo/src/core/hip/snapshot_copy.hip)** — zero-copy перенос через `hipHostGetDevicePointer` (`CopyAlignedSnapshot`), напрямую ускоряет чанковое потоковое чтение [`Restore`](ROADMAP_SPEC_01.md#L171).
* **Зрение (§2.А п.1–4) и NPU (§3.3):**
  * В `Q6dense` (`EXPERIMENTS.md` #16) пер-строчный диагональный PV уже реализован у нас в `attention_wmma.hip` — падающий тест `35/36` (`continued image cache differs from cold prefill`) требует нашей отладки на Этапе 1.
  * Апстрим `ce617610` (`+54/−3`, #1034 vision sidecar) отсутствует в `faex2`, но конфликтует с нашим fused-attention (как и `d8138a01`, указано в roadmap §2.А п.4) — не тянуть вслепую до починки `35/36`.
  * Апстрим `649824af` (защита non-finite logits после NPU-prefill) остаётся в Этапе 5 до исправления бага Data Fabric MC28.

---

## 9. Полная карта переносимой логики `strixite` (AGPL-3.0: только идеи, 0 строк кода!)

При первичном анализе идеи из `_borrow-eval/strixite` были учтены фрагментарно (только 4 заметки по WMMA/layout/kv8). Ниже — **полный аудит всех модулей `strixite` (`serve/` и `kernels/`)**, которые полезны для `gufo-faex2` как **чистые инженерные идеи**, реализуемые с нуля по нашему дизайну (ни одной строки кода из AGPL-репозитория не копировать и не переводить построчно!).

### 9.1. Кэш продолжений, Hermes Agent и сжатие истории (`ROADMAP` §1.2 п.6, §2.В п.11)
| Модуль `strixite` | Суть идеи | Какую проблему `gufo-faex2` решает и куда ложится |
| :--- | :--- | :--- |
| **`ram_only` для one-shot запросов** ([prompt_cache.hpp:41-45](../../../_borrow-eval/strixite/serve/prompt_cache.hpp#L41-L45)) | Запросы в 1 сообщение без `tools` и продолжений держатся в RAM для повторов, но **никогда не пишутся на диск** и не вытесняют длинные сеансы (в `strixite` это убрало ~230 ГБ/день лишней записи на SSD). | Идеально для служебного `aux_title_request.json` в Hermes Agent (Этап 1 п.2, `ContinuationCache` / планировщик). |
| **`Kind::System` / `Checkpoint` / `Turn` + delta-снимки** ([prompt_cache.hpp:34-65](../../../_borrow-eval/strixite/serve/prompt_cache.hpp#L34-L65)) | Системный префикс (~55K символов Hermes) живёт отдельно как `System`; перед последним user-сообщением хранится `Checkpoint` (переживает правку/сжатие хвоста); глубокие ходы пишутся как delta-приращение от базы с idle I/O приоритетом и квотой `write_gib_per_hour`. | Защита 55K-префикса при сжатии истории Hermes и защита от раздувания staging-очереди записи (как 10,2 ГиБ в тесте `r3-earlyskip`). |
| **`resume_loss`** ([resume_loss.hpp](../../../_borrow-eval/strixite/serve/resume_loss.hpp)) | При промахе кэша глубже порога (`512` токенов, `>=8192` — WARNING) определяет по границам `<\|im_start\|>` **точное сообщение и позицию токена**, из-за которой DeltaNet-кэш не смог продолжиться. | Диагностика в логах сервера `faex2`: сразу видно, какое именно изменение клиента/сжатия вызвало холодный пересчёт длинного префикса. |
| **`GrammarCache`** ([grammar_cache.hpp](../../../_borrow-eval/strixite/serve/grammar_cache.hpp)) | LRU-кэш скомпилированных JSON-схем и масок токенов (до 16 схем; `aux_title_request.json` занимает ~190 КБ). | Убирает повторную компиляцию `response_format: json_schema` для однотипных заголовков Hermes и MCP-вызовов. |

### 9.2. Управление единой памятью 128 ГБ UMA и скорость снимков (`ROADMAP` §2.В п.9–10)
| Модуль `strixite` | Суть идеи | Какую проблему `gufo-faex2` решает и куда ложится |
| :--- | :--- | :--- |
| **`MemoryGovernor`** ([memory_governor.hpp](../../../_borrow-eval/strixite/serve/memory_governor.hpp)) | За цикл «проверка → вытеснение → аллокация» выполняется **ровно одно** чтение `/proc/meminfo` (одна точка правды для всех буферов) с жёстким несгораемым остатком `floor`. | Закрывает гонки при падении `MemAvailable` на Strix Halo UMA (паузы >1 с в `EXP-N`). |
| **`plan_memory` (`fail` / `adapt`)** ([memory_plan.hpp](../../../_borrow-eval/strixite/serve/memory_plan.hpp)) | Проверка при старте после загрузки весов: если не хватает `MemAvailable`, либо явный отказ с расчётом вместимости, либо поэтапная деградация (выкл. spare-буферов → RAM-уровня кэша → урез буферов) с `WARNING`. | Устраняет риск зависаний при нуле `MemAvailable` и заменяет молчаливые отказы понятной диагностикой старта. |
| **`HostBuffer`** ([host_buffer.hpp](../../../_borrow-eval/strixite/serve/host_buffer.hpp)) | Переиспользуемый буфер через `mmap`/`mremap` + предварительный fault страниц вместо `std::vector::resize` (в `strixite`: ~782 мс → 66 мс при повторе на 5 ГиБ). | Может заметно ускорить экспорт/импорт снимков (рядом с `StreamPersistent` и потоковым `Restore` в Этапе 3). |

### 9.3. Спекуляция, Cost Gate, планировщик и телеметрия (`ROADMAP` §1.2 п.3, п.6, Этап 1–2)
| Модуль `strixite` | Суть идеи | Какую проблему `gufo-faex2` решает и куда ложится |
| :--- | :--- | :--- |
| **`capture.hpp` (`--capture-dir`)** ([capture.hpp](../../../_borrow-eval/strixite/serve/capture.hpp)) | Опциональная запись завершённого запроса: заголовок с флагами (`tools`, `one_shot`, `mtp_margin`) + `int32` токены + 3 байта на шаг (`{drafted, accepted, emitted}`). | Даёт **телеметрию длин историй** для решения `sparse` vs `Q8_0` и позволяет **офлайн** моделировать любые пороги cost gate без запуска GPU. |
| **`MtpPick` (top-2 логитов + маржа + ошибки в 64 Б)** ([mtp_pick.hpp](../../../_borrow-eval/strixite/kernels/mtp_pick.hpp)) | За один 64-байтный copy отдаёт на CPU не только `argmax` словаря черновика, но и `second_v` (разница логитов `best_v - second_v`) + NaN + 9 слов ошибок. | Эволюция [`LaunchBatchedGPUArgmax`](../../src/models/qwen/hip/mtp/executor.cpp#L342) для cost gate (`T7`) и отсечки неуверенных черновиков (как `0.5` в `Q6dense`) без копирования всего словаря на CPU. |
| **`GpuArbiter` (`others_decoding`)** ([seq_backend.hpp](../../../_borrow-eval/strixite/serve/seq_backend.hpp)) | Контракт: черновая цепочка не прерывается другими слотами до верификации; пока активны декодеры, чужой prefill автоматически дробится на меньшие чанки. | Полезное дополнение к `SetPrefillYield` (Пакет D) и будущему приоритету коротких семейных запросов в [`text_generation_scheduler.cpp`](../../src/cli/serve/text_generation_scheduler.cpp). |

### 9.4. Квантованный KV-кэш, зрение, `<think>` и раскладка весов (`ROADMAP` §2.А п.3, §3.1, §3.2)
| Модуль `strixite` | Суть идеи | Какую проблему `gufo-faex2` решает и куда ложится |
| :--- | :--- | :--- |
| **`kv8` (поканальный K + `exact_from` + prefill warning)** ([kv8.hpp](../../../_borrow-eval/strixite/kernels/kv8.hpp)) | K квантуется **по каналам** (группа 32 строк, −42 % ошибки после RoPE), V — по блокам 32; неполный хвост `<32` строк всегда в FP16 (`kv8_exact_from(p)` = независимость результата от нарезки prefill); prefill читает staged FP16 (распаковка 8-бит внутри WMMA замедляет в **2,7×**). | Повышает точность и сохраняет кэш-идентичность при реализации условного `Q8_0` KV на Этапе 3. |
| **`vision_linear_bias` + `BicubicAntialias` + `max_patches`** ([vision.hpp](../../../_borrow-eval/strixite/kernels/vision.hpp), [vision_preprocess.hpp](../../../_borrow-eval/strixite/serve/vision_preprocess.hpp)) | Прибавление `bias` прямо к FP32-аккумулятору до округления дало 7 % точности по 27 слоям ViT; `BicubicAntialias` и жёсткий кламп `max_patches` аккуратно сжимают огромные картинки. | Ориентир для задачи §2.А п.3 (адаптивный даунскейл крупного текста 48–96 pt и стабильность зрения). |
| **`ThinkWatch` (`think_nudge`)** ([think_nudge.hpp](../../../_borrow-eval/strixite/serve/think_nudge.hpp)) | Отслеживает повторы 8-грамм в окне 1024 токенов (после 3072 токенов, порог `25 %`) и мягко подталкивает модель на границе абзаца `"\n\n"`, **принципиально не обрывая** длинное честное рассуждение по длине. | Опциональная защита от бесконечного зацикливания `<think>` в Hermes / OpenCode без ухудшения сложных ответов. |
| **`wmma_gemm` + `strixw`** | K-шаг `kWmmaKC = 64`, LDS-шаг `kWmmaLd = 72` (+8 против bank conflicts), выравнивание `K % 64 = 0`. | Архитектурный референс, если в будущем разморозим `.ghalo` (§3.1). |

**Что из `strixite` НЕ подходит даже как идея:** `qsa` (`qwen38_flash_next` — у 35B-A3B нет QSA-индексатора), 512-экспертный MoE-роутинг, любые прямые фрагменты кода (AGPL-3.0).

---

## 10. Правовой аспект заимствований (подтверждено полностью)

| Источник | Лицензия | Сторонний код внутри | Правило для `gufo-faex2` |
| :--- | :--- | :--- | :--- |
| **`gufo-laguna`** | **MIT** (в нашем дереве) | общий `mmq` (MIT) | Свободно в пределах MIT. |
| **апстрим `gufo`** | **MIT** | `THIRD_PARTY_NOTICES.md` | Штатный перенос патчей. |
| **`tubo`** | **MIT** | `llama-cpp-turboquant` (MIT); `AMD IRON pack_b` (Apache-2.0) | Код переносим с сохранением заголовков, нотисов в `NOTICE`/`THIRD_PARTY_NOTICES.md` и пина `VENDOR.md`. |
| **`Q6dense`** | **MIT** | `llama.cpp` `5c0e9468` (MIT, `VENDOR.md`) | Код (`qwen36_35b_a3b`) переносим с сохранением нотисов MIT. |
| **`strixite`** | **GNU AGPL-3.0** | `written from scratch` | **Копировать код категорически запрещено** (AGPL §13 на сетевой `gufo serve`). Переносим только идеи (§9) с написанием кода с нуля. |
| **Веса HF** | Отдельно от движка | Индивидуально по карточке HF | Проверять лицензии GGUF-карточек перед скачиванием на стенд. |

---

## 11. Что заимствовать и делать НЕ нужно

1. **Замена всего нашего движка на `qwen36_35b_a3b` из `Q6dense` или `PrefillQ8` дубликаты весов:** теряем multi-user continuation cache и память (но *точечные* MIT-решения под 35B — пул графов с `TrimRollback`, 8-бит KV, `DenseQ6KVec`, `MtpHiddenKernel` проверить можно и нужно).
2. **Апстримные `kMtpCycleMilliseconds` (`qwen38_flash_next`):** другие модель и профили; для 35B-A3B используем собственную калибровку.
3. **`turbo4_attention.hip` целиком (4256 строк из `tubo`):** написано под 27B и конфликтует с нашей историей `attention_decode.hip`; берём только уроки (§5.3), `MEMTRIM_9` и `snapshot_copy.hip`.
4. **In-kernel MoE grouping (`Q6dense` #25) и слитное norm+quant `1b1dfa80`:** уже доказанно медленнее или ломают побитовую точность (§1.3 roadmap).
5. **Код `strixite` (AGPL-3.0), `QSA`, `pi` и `4 × 262144` в рабочий пресет.**

---

## 12. Итоговая синхронизация с этапами ROADMAP_SPEC_01

| Этап ROADMAP_SPEC_01 | Согласованное содержимое с учётом аудита |
| :--- | :--- |
| **Этап 0** | Без изменений: отсев `DFlash2 vs MTP1` на `hermes_synth` + закрытие долга `004d8fa1`. |
| **Этап 1** | Телеметрия длин (идея `capture` `strixite`), 50–100 задач, приоритеты в `text_generation_scheduler.cpp`, защита кэша от `aux_title_request.json` и сжатия (`ram_only`/`System+Checkpoint`/`resume_loss` идеи `strixite` + апстрим `fe2694bc`/`8ab834a5`), починка зрения `35/36` и `Q1`, Пакет B `/tokenize` + `input_tokens` (`T1–T2`). |
| **Этап 2** | **1-м шагом:** `GUFO_QWEN_SPARSE_ATTN=1` (`T15`, 0 строк кода, порог `~+26 %` в MTP1). **Затем:** бакетные HIP-графы + `wave64` device-pos + `ForwardVerificationBatch`/`MTP`/`C4` + прогрев `hipBLASLt` (`T3–T5`, ~600–1200 строк); диагностика `MTP2/3` / grouped `down_exps`; 35B cost gate (`T7`, вкл. при провале DFlash2). |
| **Этап 3** | **Потоковое чтение `Restore`** (`snapshot_copy.hip` из `tubo` + `HostBuffer`-идея) и **разбор зазора памяти `+4..7,5 ГиБ`** (`GUFO_DEV_MEMTRIM_9` из `tubo` + `MemoryGovernor`-идея). **`Q8_0` KV (`T8–T12`) — строго условно**, только при провале `sparse` на Этапе 2 и частых >32K по телеметрии! (`kTurbo6` — далёкий резерв). |
| **Этап 4** | Эксперимент GGUF `Q6dense` (`T14`, 0 строк) + при пользе точечная оценка MIT-оптимизаций `Q6dense` (`T14b`); `.ghalo` отложено. |
| **Этап 5** | Без изменений: NPU при закрытии бага MC28 (+ `649824af`). |

Подробный пошаговый план задач — [BORROW_PLAN_2026-11-17.md](BORROW_PLAN_2026-11-17.md).
