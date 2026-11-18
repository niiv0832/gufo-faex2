# ROADMAP_SPEC_01: Единый план доработок, проверок и техническая спецификация `gufo-faex2`

Дата: 2026-11-18
Статус: **Рабочая спецификация (ревизия 4 — синхронизировано по итогам `T1`, `T2`, `T2a-1`, `T2a-2`, выката `b81812aa` в прод и часового прогона памяти)**
Текущий рабочий продакшен: **`b81812aa`** (бинарник `e5af3a93…`, резервный откат — **`be242d70`** / `964e84e5…`, ядро `7.3.4-1-cachyos`).
Целевая модель: **Qwen3.6-35B-A3B** (40 слоёв: 30 *Gated DeltaNet* + 10 *Gated Attention*, `16 Q-heads / 2 KV-heads`, `hidden = 2048`; 256 роутируемых экспертов, активны 8 + 1 shared, `expert_ff = 512`) + встроенный **MTP** *(рабочий режим `MTP1`; `DFlash2` отклонён на Шаге 1 2026-11-17)* + **Vision Encoder** (`mmproj-BF16.gguf`).  
Целевое железо: **128 ГБ UMA, AMD `gfx1151`** (стенд FEVM FAEX2; перспектива — Gorgon Halo / исправленный XDNA2 NPU).  
Клиенты: основной — **Hermes Agent** (семейный чат, русский язык, инструменты, зрение); второй — **OpenCode 2.3.0** (программирование).

> **Зачем нужен этот документ (простыми словами):**  
> Этот документ собирает в одну инженерную спецификацию **три потока задач**, которые раньше лежали в десятке разных отчётов:
> 1. **Наш собственный техдолг и открытые находки аудита** ([AUDIT_CLOSURE_ROUND3](AUDIT_CLOSURE_ROUND3_2026-11-14.md), [REMEDIATION](REMEDIATION_2026-11-13.md), [INVESTIGATION_RESULT](INVESTIGATION_RESULT.md), [SNAPSHOT_STREAMING_PLAN](SNAPSHOT_STREAMING_PLAN.md)).
> 2. **Наши собственные планы сравнения и переноса из апстрима** ([DFLASH2_EVAL_PLAN](DFLASH2_EVAL_PLAN_2026-11-15.md), [UPSTREAM_PORT](UPSTREAM_PORT.md), [OPTIMIZATION_BACKLOG](OPTIMIZATION_BACKLOG.md), [DECODE_DEPTH](DECODE_DEPTH.md), [NPU_MAP](NPU_MAP.md)).
> 3. **Лучшие архитектурные идеи из внешних/соседних проектов** (`strixite`, `gufo-laguna`, `tubo`, `NinjaPear/Q6dense`, апстрим `gufo`) с проверкой на отсутствие конфликтов.

---

## 1. Карта проектов и заимствований: что уже есть, что проверяем, что переносим, что не трогаем

### 1.1. Что уже сделано в `gufo-faex2` (повторно не изобретать и не переносить)

По результатам [OPTIMIZATION_BACKLOG.md](OPTIMIZATION_BACKLOG.md), [DECODE_DEPTH.md](DECODE_DEPTH.md) и пакетов A/C/D из [UPSTREAM_PORT.md](UPSTREAM_PORT.md) в ветке **уже работают**:
* **Внимание основной модели на длинной истории (`6bcf0c3b`):** wave64 split-K ядро для геометрии `16 голов запроса на 2 головы памяти` (читает KV-плитку один раз на группу из 8 голов) → **+20 %** на истории 64K токенов (~199 ГБ/с).
* **Послойные уступки prefill рядом с пишущими соседями (Пакет D, `21736dd0`, `afcff8fb`):** `SetPrefillYield` и `PrefillLayerCheckpoint` сохраняют эффективную ширину прохода 4096 токенов, уступая декодерам между группами слоёв → чтение **+43 %**, соседу **+42 %**.
* **Веса в памяти устройства и обход page cache для снимков (Пакет C, `00eddb85`):** анонимная память процесса снижена с 22,5 до 0,19 ГиБ, снимки кэша не вытесняют память GPU.
* **Ядра K-квантов под LLVM 24 (Пакет A, `92239590`):** фикс spill в IQ4 small-batch, GEMV «строка на warp», маскированный 16-строчный prefill GEMM → короткие дозагрузки длинной истории **+9–22 %**.
* **Оптимизации MTP1 для 35B-A3B:** память MTP-головы в `FP16` (−2 ГиБ, +11,6..+19 % на 64K–120K, `69aa1d05`), сокращённый словарь 161 074 строки (латиница + кириллица + знаки, +3,9 %, `be728559`), общий проход проекции на словарь для нескольких пользователей (`GUFO_MTP_MOE_MAX_COHORT=4`, +7..+10 % на 2–4 потоках, `2da32906`, `1195d154`), потоковая запись снимков Qwen+MTP через буфер 8 МиБ (`ffc8580a`).
* **Изоляция рабочих буферов prefill по потоку GPU (`f3e8fc32`, `67007f07`, `d891354f`):** закрыта гонка между сеансами в `LaunchBatchedQuantGEMM`, `Qwen35MoePrefill` и `QwenMoeScratch` (в [`executor.hpp`](../../src/models/qwen/hip/executor.hpp#L320), историческое имя `GetMoeFfnScratch`; +~160 МиБ на сеанс).
* **API токенизации (`T1`, `T2`, 2-я очередь Пакета B):** вручную перенесены из апстрима `POST /tokenize` (`6d793814`), `POST /v1/chat/completions/input_tokens` и `POST /v1/responses/input_tokens` (`4ab04ab8`).
* **Изоляция служебных запросов и телеметрия кэша (`T2a-1`, в проде `be242d70`):** политика `ram_only` (`<= 1024` токенов для structured one-shot без `tools`/истории), гистограммы/счётчики `/metrics` (`gufo_request_prompt_tokens`, `gufo_request_cached_fraction`, `gufo_request_cache_total`, `gufo_active_long_requests`) и диагностика потерь префикса `cache_prefix_loss`.
* **Планировщик, системный префикс, динамический потолок RAM и `GrammarCache` (`T2a-2`, ревью `APPROVE` и стендовая проверка `b81812aa` PASS):** приоритет коротких запросов `<= 2048` (`max_short_bypasses = 4`) в `PopQueued` и адаптация чанка при `others_decoding` (`min_decode_active_tokens = 128`, без ломки `SetPrefillYield`); одноразовая токенизация и сохранение `system_prefix_tokens >= 512` (`kBranchPoint` при сжатии истории) и повышение `IsBranchPoint` при `exact_replacement` (`fe2694bc`); единое чтение `/proc/meminfo` + учёт удерживаемых снимков в `ReserveSnapshot` (`MemoryGovernor`); кэши грамматик переведены на LRU (`ToolParameters = 64`, `BindConstraint = 16`); удалена недостижимая ветка `767c6018` в `StreamPersistentSnapshot`.

---

### 1.2. Что изучаем и проверяем в существующих реализациях (`strixite`, `gufo-laguna`, `tubo`, `NinjaPear`, апстрим `gufo`)

| Область | Где смотреть | Что у нас сейчас | Что именно проверяем / дорабатываем |
| --- | --- | --- | --- |
| **1. Бакетные HIP-графы (`hipGraphExec_t`)** | `gufo-laguna` ([`src/models/laguna/`](../../src/models/laguna/gpu/engine.cpp)), `Q6dense` (MIT); *(в `strixite` графов 0)* | [`HipGraphDecodeExecutor`](../../src/core/hip/detail/hip_graph_decode_executor.hpp) держит один `instance_` только в [`ForwardToken`](../../src/models/qwen/hip/decode.cpp#L50-L108) (одиночный AR `T=1`) и отключается при `kSplitKAttentionRequired` и `kLogitsNotRequested` ([`execution_policy.hpp`](../../src/models/qwen/hip/execution_policy.hpp)). При `MTP1` графов нет вовсе в [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564), [`ForwardTokenBatch`](../../src/models/qwen/hip/batched_decode.cpp#L236) и [`QwenMtpGpuExecutor::Run`](../../src/models/qwen/hip/mtp/executor.cpp#L232). | Пул графов под длины верификации и бакеты, device-`pos` в обоих split-K ядрах ([`attention_decode_verify_wave64.hip`](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip) с порогом `deep = start_pos + 1 >= 512` и [`attention_decode.hip`](../../src/models/qwen/hip/kernels/attention_decode.hip)), захват `ForwardVerificationBatch` + `QwenMtpGpuExecutor`, разделение `C1` и `C4` (массив per-session позиций), инвалидация графов при `TrimRollback` (урок `Q6dense`) и стартовый прогрев `hipBLASLt`. |
| **2. MoE-роутинг, grouped/indirect GEMM и просадка `MTP2/3`** | `qfn_mmq` ([`moe_vec.hpp`](../../src/models/qwen/hip/ops/moe_vec.hpp)), `Q6dense` (MIT); *(сторонний роутинг из `laguna`/`strixite`/`tubo` не заимствуем)* | FP32-роутер и `LaunchMoeRouterTopK` ([`moe.hip`](../../src/models/qwen/hip/kernels/moe.hip)); grouped/indirect `gate/up` GEMM и [`QwenMoeScratch`](../../src/models/qwen/hip/executor.hpp#L320) уже работают. Однако при `MTP2`/`MTP3` на 2–4 потоках скорость падает на **5–33 %** ([CONTEXT](CONTEXT.md)), а условие `by_expert` в [`qfn_mmq.hip.cpp`](../../src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp#L505-L508) (`K == 640 && (Q5_1 \|\| Q8_0)`) выключено для `down_exps` на 35B-A3B (`K = 512`, `Q4_K/Q5_K/Q6_K`). | **Закрыто:** переписывание роутинга с нуля не нужно. **Остаётся открытым:** профилирование просадки `MTP2/3` (включая отсутствие графов на `MTP` и путь внимания внутри `MTP`-головы) и проверка группировки `by_expert` для `down_exps` при `K = 512` (`Q4_K/Q5_K/Q6_K`, образец `Q6dense` MIT). |
| **3. Cost gate спекуляции (`MTP1 ↔ AR`)** | Апстрим `gufo` (`1167a944` для 27B, `2caf2a82` уже учтён), `Q6dense` (MIT, порог `0.5`), `strixite` (AGPL — только идеи `capture`/`MtpPick`); *(в `gufo-laguna` гейта нет)* | [`QwenAutoDraftBackend`](../../src/models/qwen/hip/auto_draft.cpp) смотрит только на `tokens_per_second`, а порог когорты MTP жёсткий (`GUFO_MTP_MOE_MAX_COHORT=4` в [`mtp/backend.cpp`](../../src/models/qwen/hip/mtp/backend.cpp#L265-L292)). **По итогам Шага 1 ([DFLASH2_STEP1_REPORT_2026-11-17.md](DFLASH2_STEP1_REPORT_2026-11-17.md)) `DFlash2` закрыт для `35B-A3B`.** | В `T7 cost gate` остаётся **только ветка `MTP1 ↔ AR` на `C2–C8`** с учётом активных потоков, acceptance rate, длины контекста, отсечки неуверенных черновиков (`kMinDraftProbability = 0.5`, идея `Q6dense`), маржи top-2 (идея `strixite::MtpPick`) и офлайн-калибровки на телеметрии (идея `strixite::capture`). Ветка `MTP1 → DFlash2` снята. |
| **4. Ускорение длинного контекста: `GUFO_QWEN_SPARSE_ATTN=1` vs условный `Q8_0` KV-кэш** | Собственный [`attention_sparse.hip`](../../src/models/qwen/hip/kernels/attention_sparse.hip) (`34e61a15`); `Q6dense` (MIT), `tubo` (MIT — только уроки), `strixite` (AGPL — только идея `kv8`) | KV-кэш 10 слоёв Attention хранится в `FP16` (**10,00 ГиБ / 10,74 ГБ** на `4 × 128K`). Готовое ядро `QwenSparseDecodePartialsKernel<8>` (`GUFO_QWEN_SPARSE_ATTN=1`, **0 строк нового кода**) на `MTP1` даёт ориентир **~+26 %** (`25,7 → 32,4 tok/s`; `~43 tok/s` было в `AR + NPU`). | **Первым на Этапе 2:** проверка уже готового `GUFO_QWEN_SPARSE_ATTN=1` с `MTP1` на семантику и качество (набор 50–100 задач). **Строго условно на Этапе 3 (только при отклонении `sparse` и частых >32K по телеметрии):** `Q8_0` KV (`5,31 ГиБ / 5,70 ГБ`, экономия **4,69 ГиБ / 5,03 ГБ**) с уроками `tubo`, базой ядер `Q6dense` и идеями `strixite::kv8`. |
| **5. Смесь `Q6_K` dense + `UD-Q4_K_XL` experts (`Q6dense`) и `.ghalo` (отложено)** | `Q6dense` (MIT — рецепт GGUF и ядра `qwen36_35b_a3b`), `strixite` (AGPL — только идеи `strixw`/`wmma`) | Стандартный GGUF `UD-Q4_K_XL`. Опыт `Q6dense` показал, что ускорение даёт перепаковка тензоров *при загрузке в память*, а не собственный дисковый формат. | Проверить лицензии HF и `q6dense.types`, сравнить GGUF-рецепт `Q6dense` на `Q1` и скорости (`T14`, 0 строк кода) и при необходимости точечно оценить MIT-ядра `DenseQ6KVec`/`MtpHiddenKernel` (`T14b`). **Формат `.ghalo` отложен.** |
| **6. API, планировщик, телеметрия и кэш под профиль Hermes Agent** | Апстрим `gufo` (`6d793814`, `4ab04ab8`, аудит `fe2694bc`/`8ab834a5`), `strixite` (AGPL — только идеи: `capture`, `ram_only`, `System/Checkpoint`, `resume_loss`, `GrammarCache`, `others_decoding`, `MemoryGovernor`) | `T1`, `T2`, `T2a-1` и **`T2a-2`** **в проде (`b81812aa`/`e5af3a93…`, резерв `be242d70`)**: приоритет коротких, `others_decoding`, `system_prefix_tokens`, динамический `MemoryGovernor`, `fe2694bc` branch-point, `GrammarCache` LRU `64`/`16`; пройдены ревью `APPROVE`, CPU `56/56`, ASan `9/9`, Qwen GPU `77/77`, `qwen_snapshot_streaming_test`, `qwen_layered_prefill_test`, живой `--mmproj` (`images` 4/4), matched pair + smoke `7/7` и час памяти **PASS**. | Накопление статистики долей >32K–64K на живом сервере; пошаговый `--capture-dir` вынесен в подготовку `T7` (Этап 2). |

---

### 1.3. Что проверено и НЕ нужно делать (отрицательные результаты и жёсткие границы)

Чтобы не тратить окна стенда на уже опровергнутые идеи (см. [OPTIMIZATION_BACKLOG.md](OPTIMIZATION_BACKLOG.md), [UPSTREAM_PORT.md](UPSTREAM_PORT.md), [NPU_MAP.md](NPU_MAP.md)):
1. **Слитное ядро «норма + квантование активаций» перед MoE (`1b1dfa80`) — НЕ ДЕЛАТЬ:** уже собиралось и измерялось, даёт **−0,7 %** без MTP и **−0,5 %** с MTP, при этом меняет округление и ломает побитовую сходимость текстов (отменено в `fbfa0369`).
2. **Подбор геометрий small-batch GEMM для 3–8 строк (бывший п. 1 Пакета E) — НЕ ДЕЛАТЬ:** замер на `2da32906` показал, что доля малых матриц при 4 пользователях всего 4,6 %, а теоретический потолок выигрыша по всему шагу — ~1,4 %.
3. **Перенос NPU-стека из `gufo-laguna` — НЕ НУЖЕН:** по [NPU_MAP.md](NPU_MAP.md) весь NPU-стек для 35B-A3B (`1c00f953`, `35dd096c`, `7093a45e`, `736a9f4c` в `tools/npu/`) **уже находится в нашей ветке** от базы `6a107cf6`. Он выключен намеренно из-за аппаратного зависания Strix Halo Data Fabric ([NPU_FREEZE.md](../NPU_FREEZE.md)).
4. **Переезд Qwen на `src/cache` (component cache `da5d5a2b`) — ОТЛОЖЕНО:** меняет ~1000 строк `inference_backend.cpp`, адаптера для MoE нет; возвращаться только если старый `ContinuationCache` не справится с реальной нагрузкой 4 агентов Hermes.
5. **Специализация дерева моделей (S1/S2) и профиль `4 × 262144` — ЗА СКОУПОМ / ОТЛОЖЕНО:** вырезание других моделей из дерева (S2) превращает проект в жёсткий несовместимый с апстримом форк без выигрыша в скорости на 128 ГБ; профиль `4 × 262144` в рабочий пресет не выводится, так как Hermes Agent сам сжимает историю диалога. Также за скоупом: `IQ3_XXS`, агент `pi`, Rust-зависимость (`c2292921`).

---

## 2. Полный реестр собственного техдолга и нерешённых проблем `gufo-faex2`

Сводка всех открытых вопросов из [CONTEXT.md](CONTEXT.md), [AUDIT_CLOSURE_ROUND3_2026-11-14.md](AUDIT_CLOSURE_ROUND3_2026-11-14.md), [CANDIDATE_VARIANTB_2026-11-14.json](CANDIDATE_VARIANTB_2026-11-14.json), [INVESTIGATION_RESULT.md](INVESTIGATION_RESULT.md) и [SNAPSHOT_STREAMING_PLAN.md](SNAPSHOT_STREAMING_PLAN.md):

### А. Качество, точность и зрение
1. **Падающий тест зрения `model-vision-serving-mtp` (код 1, тесты 35/36):** отказ `continued image cache differs from cold prefill` (находки аудита **№9** и **№12**). **Диагностировано точными измерениями 2026-11-19 ([VISION_CONTINUATION_PHASE1_2026-11-18.md](VISION_CONTINUATION_PHASE1_2026-11-18.md) §6, окно 60 с): дефекта движка нет** — при одинаковом входном контексте на шагах `0..3` расхождение между decode-историей кэша и матричным prefill составляет `common_max_logit_delta = 0.513805` (по шагам `0..3`: `0.513805`, `0.332067`, `0.312154`, `0.391665` — в пределах `0.09–0.82` из §В `INVESTIGATION_RESULT`), `common_max_KL_temperature1 = 0.001963`; на шаге `3` жадный `argmax` переворачивается на почти равной паре `11` («,», `22.374062` vs `22.281139`, зазор `0.030472`) / `1892` («—», `22.343590` vs `22.351360`, зазор `0.070221`). Все оракулы снимков/истории точные, `cached_trajectory_matches_serving=1`, `cold_trajectory_matches_serving=1`. Проверка не ослаблялась; судьба `36/36` — вместе с контрактом `Q1` (решение владельца 2026-11-18).
2. **Неутверждённый контракт точности приближённого prefill (`Q1 DRAFT`, находка №9):**
   * На коротких пакетах (~300 токенов) KL достигал `0,055`; в target MTP R2 на RU P15/C1 есть устойчивое расхождение с AR (`KL 0,006018`, `maxΔ 1,291648`), а в greedy тексты MTP и `off` расходятся в 21 из 40 заданий (логиты `VERIFY_ROWS` отличаются от decode до `0,82`).
   * Расчёт стоимости (этап 8 третьего круга) показал, что допуск черновика (`0,001`) строже шума самой арифметики (`0,003–0,005`), а верхние длинные срезы (128 текстовых / 512 vision) недостаточны для task-допуска `0,5 п.п.` (требуется пересмотр допусков/области либо выборка ~736–2300 документов).
3. **Нечитаемость крупного текста на изображениях (48–96 pt, масштаб 8–16):** свойство весов/проектора (совпадает в Gufo и llama.cpp), но уменьшение изображения в 2–3 раза делает надписи читаемыми ([DESIGN.md](DESIGN.md) §4, [INVESTIGATION_RESULT.md](INVESTIGATION_RESULT.md) §Б). В движке нет опционального адаптивного даунскейла; кроме того, для картинок ≤320×96 минимальный размер сетки токенов у Gufo отличается от llama.cpp (106 против 61).
4. **Непроверенные комбинации зрения, выборки и апстрим-правок:**
   * `d8138a01` (`detail: low`) и `ce617610` (sidecar при перезапуске) конфликтуют с нашим fused-attention; `GUFO_VISION_ATTENTION=tiled` не проходил отдельную квалификацию кэш-идентичности (находка **I3**); `fe9498d1` (GCC 16 `<format>`) требует сверки с нашим `5409fdd9` при пересборке *(примечание: `f948ddd3` уже перенесён в ветке, см. [UPSTREAM_PORT.md](UPSTREAM_PORT.md) §2)*.
   * Не завершена широкая статистическая квалификация `sampled-MTP` при переходах нагрузки, мультиизображениях и одновременной работе зрения со спекуляцией и чужим длинным prefill ([BASELINE.md](BASELINE.md)).
   * **Практический набор оценок качества семейного чата (50–100 фиксированных примеров):** контракт `Q1` проверяет численную близость prefill к `G1`, но без собственного набора прикладных задач (русские инструкции, школа, OCR/таблицы, цепочки инструментов Hermes) любое приближение (`GUFO_QWEN_SPARSE_ATTN`, `Q8_0` KV, `Q6_K`) нельзя оценить на реальную полезность ответов.

### Б. Спекуляция (MTP / DFlash2) и нагрузка Hermes Agent
5. **Оценка DFlash2 против MTP1 завершена на Шаге 1 ([DFLASH2_EVAL_PLAN_2026-11-15.md](DFLASH2_EVAL_PLAN_2026-11-15.md), [DFLASH2_STEP1_REPORT_2026-11-17.md](DFLASH2_STEP1_REPORT_2026-11-17.md)):**
   * Шаг 0 выполнен (15 реальных запросов Hermes Agent v0.22.2 в `tests/functional/fixtures/hermes_synth/` + `aux_title_request.json`).
   * **Шаг 1 выполнен 2026-11-17 на стенде `faex2` (`004d8fa1`, `C1`, `temperature = 0`):** на классе `tool` `DFlash2` оказался медленнее `MTP1` (−8,76 % в Проходе 1, −16,13 % в Проходе 2, −13,86 % по `decode tok/s`), а по всему эпизоду медленнее на −10,95 % / −13,00 % при принятии черновиков 48,7–52,7 % против 83,2–91,8 % у `MTP1`. По критерию отказа **тема `DFlash2` для `Qwen3.6-35B-A3B` закрыта окончательно, Шаг 2 не нужен**.
   * Подтверждённые особенности Hermes (влияют на Этап 1): **(а)** постоянный префикс ~55 тыс. символов (~12 288 токенов) попадает в кэш со второго запроса, сокращая `TTFT` на `MTP1` с 4899 до 151–529 мс; **(б)** `reasoning_effort: medium` (рассуждение + JSON); **(в)** служебный запрос заголовка `aux_title_request.json` (`response_format: json_schema`, `reasoning_effort: none`) отработал валидно и не вытеснил 12K-префикс в `C1`.
6. **~~Условный техдолг DFlash2~~ (СНЯТ 2026-11-17 по итогам Шага 1):**
   * Потоковая запись снимков (`StreamPersistent` с лимитом ≤16 MiB) для `QwenDFlashDraftSnapshot` / `QwenAutoDraftBackend` и калибровка таблиц `dflash_policy.hpp` для sampled/mixed пакетов больше не требуются (остаются историческим ограничением неактивного режима `dflash2`).
7. **Просадка `MTP2`/`MTP3` при 2–4 потоках (−5…−33 % на `004d8fa1`):** отсутствуют HIP-графы на шаге MTP, есть собственное чтение KV внутри MTP-головы ([DECODE_DEPTH.md](DECODE_DEPTH.md)), а в [`qfn_mmq.hip.cpp`](../../src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp#L505-L508) не срабатывает группировка `by_expert` на `down_exps` при `K = 512` и `Q4_K/Q5_K/Q6_K`.

### В. Кэш, память, планировщик и эксплуатация
8. **Статус закрытия проверок рабочего продакшена `be242d70` и резервного `004d8fa1`:**
   * Часовой прогон памяти (`compose --duration 3600`, ядро `7.3.4-1-cachyos`) и 4-пользовательский профиль **ЗАКРЫТЫ на `be242d70` (2026-11-18, оба PASS)**; тем самым долг `004d8fa1` закрыт по содержанию (оговорка «час ≠ 24/7» сохраняется).
   * Недостижимая защитная проверка `qwen_snapshot == nullptr` из `767c6018` в `StreamPersistentSnapshot` **удалена** (`4f934957`).
   * **Остаётся открытым:** независимое ревью коммитов варианта Б (`158711ad`, `004d8fa1`) и новых коммитов `T2a-2`; 8 ячеек `UNMATCHED_WORK` в `r3-speed` (+85 байт в файле снимка).
9. **Потоковое чтение снимков (`Restore` / `ReadImage`) и буферы потоков GPU:**
   * Запись снимка стала потоковой (scratch 8 MiB), но **восстановление с диска (`Restore`) по-прежнему читает весь многогигабайтный файл снимка в RAM целиком** ([DESIGN.md](DESIGN.md) §2б, [SNAPSHOT_STREAMING_PLAN.md](SNAPSHOT_STREAMING_PLAN.md)).
   * Пер-поточные буферы GPU (`d891354f`, ~160 МиБ на сеанс) живут до конца процесса и не освобождаются при уничтожении потока HIP.
10. **Редкие аномалии задержек и стабильности:**
    * **Двухрежимная пауза очереди при 4 пользователях (C4) и на первом запросе:** всплески 27–39 мс против 1 мс на `004d8fa1` и `abbfb006` (связаны с приходом запроса после старта неполной первой группы `codex-diagnostic2` и ленивой загрузкой ELF-модулей `hipBLASLt` 3,6 МиБ).
    * **Два старых полных промаха дискового кэша (`final2` / рабочий `1195d154`, 2 из 736):** причина отказа диска (вероятно «занято») до конца не доказана; решено наблюдать в эксплуатации по новым логам причин (`fb816341`, `158711ad`).
    * **Очередь записи при стресс-тесте малого дискового кэша (16 ГиБ против штатных 64 ГиБ):** в диагностическом прогоне `r3-earlyskip` (где объём дискового кэша намеренно зажимался до 16 ГиБ для провокации постоянного LRU-вытеснения) `staging_used_bytes` вырастал до ~10,2 ГиБ даже с правилом раннего отказа `min_step` (на штатных 64 ГиБ с правилом — 0 ГиБ).
    * **Паузы под давлением памяти (`EXP-N`, Пакет C):** до 6 пауз >1 с в первые 40 с под искусственным давлением.
    * **Единичное зависание MTP-сервера (выход с кодом 75 по таймауту 120 с на `r138` из ~3800 запросов, [INVESTIGATION_RESULT.md](INVESTIGATION_RESULT.md) §Д):** причина не установлена; автоперезапуск `gufo-faex2.service` после реальной аварии и реальной перезагрузки машины ещё не проверен в бою; нет квалификации 24/7.
    * **Устаревшая документация модели и клиентские памятки:** в [`docs/models/qwen3.6-35b-a3b/README.md`](../models/qwen3.6-35b-a3b/README.md) описание vision актуализировано (2026-11-17); на клиентской стороне OpenCode (вне репозитория движка) остаётся открытым вопрос глобального правила русского языка в `~/.config/opencode/AGENTS.md`.
11. **Телеметрия реальной нагрузки, приоритеты планировщика и кэш-политика под Hermes Agent (План Б из [DFLASH2_EVAL_PLAN](DFLASH2_EVAL_PLAN_2026-11-15.md#L57)):**
    * **Распределение длин историй на рабочем сервере:** гистограммы и счётчики `/metrics` (`T2a-1`) уже развёрнуты в проде `be242d70`; остаётся накопить эксплуатационную статистику долей историй >32K–64K токенов.
    * **Приоритизация очереди планировщика и защита 55K-префикса (`T2a-1` в проде, код `T2a-2` реализован 2026-11-18):** `ram_only`, `cache_prefix_loss`, приоритет коротких запросов `<= 2048` (`max_short_bypasses = 4`), `others_decoding`, `system_prefix_tokens >= 512`, `fe2694bc` `IsBranchPoint`, `MemoryGovernor` и `GrammarCache` LRU (`ToolParameters = 64`, `BindConstraint = 16`) реализованы в коде и проверены локально в CPU/ASan-UBSan тестах; остаются стендовая HIP-сборка и живая проверка `T2a-2` на стенде. Пошаговый `--capture-dir` вынесен к задаче `T7`.
    * **Неквалифицированные зазоры:** временная цена `late_shared_prefix` (`wait=0`, находки **№3** и **I7**), хотя сам сценарий проверен функционально по HTTP в `gates2`; сравнение скорости с `llama.cpp` после `6bcf0c3b` и `T2a` и разбор зазора памяти (+4–7,5 ГиБ к `llama.cpp`, [CONTEXT.md](CONTEXT.md#L68)).

---

## 3. Техническая спецификация целевых доработок

### 3.1. Бинарный формат весов (`.ghalo` — ОТЛОЖЕНО; приоритет у проверки `Q6dense` при загрузке)

> **Статус (решение по `T16` / `BORROW_EVAL_2026-11-17.md` §6):** собственный дисковый контейнер **`.ghalo` отложен**. Опыт `Q6dense` (MIT) показывает, что основной прирост на 64K–131K дают GGUF-схема точности (`Q6_K` dense + `UD-Q4_K_XL` experts) и перепаковка/раскладка весов *при загрузке в память*, а не новый формат файла. Архитектурная спецификация ниже сохранена как справочный задел.

Текущий движок читает стандартный GGUF (`UD-Q4_K_XL`), перекодируя при загрузке только отдельные тензоры (например, `Q6_K` → `Q8_0` в части путей и `BF16` → `Q8_0` в DFlash2). Для будущего сравнения смешанной точности без потери совместимости с GGUF сначала проверяется рецепт `Q6dense` (Этап 4); ниже приведена сохранённая спецификация опционального предвыровненного образа **`.ghalo`**.

#### Смешанная схема точности
* **Router (`ffn_gate_inp`) и `shared_expert` gate:** строго **`FP32` или `BF16`** (квантованный роутер запрещён коммитом `355b1d86`, логиты роутера считаются в `FP32`).
* **Dense-проекции (`QKV`, `attn_out`, `ssm_in`, `ssm_out`) и тело `shared_expert`:** **`Q6_K`** (или `Q8_0` для путей с 8-битными активациями).
* **256 роутируемых экспертов (`gate`, `up`, `down`, `expert_ff = 512`):** **`UD-Q4_K_XL`**.
* **Vision Encoder (`mmproj`):** **`BF16` / `FP16`**.

#### Структура заголовка `.ghalo` (128 байт, Little-Endian)
```text
Offset  Size  Field              Description
0x00    8     magic              "GHALO001"
0x08    4     version            uint32 (1)
0x0C    4     arch_id            uint32 (1 = Qwen3.6-35B-A3B)
0x10    8     flags              bit0: has_npu_tiles, bit1: has_vision, bit2: has_mtp, bit3: has_dflash
0x18    8     tensor_count       uint64
0x20    8     toc_offset         uint64 (выровнено по 64 байтам)
0x28    32    source_sha256      SHA-256 исходного GGUF/SafeTensors
0x48    32    numerics_id_hash   SHA-256 численной политики и раскладки (для контроля кэша, находка I3)
0x68    24    reserved           Нули (по аналогии с outer112B/compact80B)
```
* **Запись таблицы тензоров (`TensorEntry`, ровно 64 байта, Little-Endian, естественное выравнивание 8-байтовых полей в начале без скрытого паддинга):**
  ```text
  Offset  Size  Field         Description
  0x00    8     name_hash     uint64: FNV-1a 64-bit имени тензора (тот же полином, что в execution_policy.hpp)
  0x08    8     data_offset   uint64: смещение данных в файле, кратно 128 байтам (под cache-line и float4/uint4)
  0x10    8     byte_size     uint64: размер тензора в байтах
  0x18    8     checksum      uint64: FNV-1a 64-bit (или младшие 64 бита SHA-256) от байтов тензора
  0x20    4     dtype         uint32: тип квантования/элемента (GgmlType / раскладка)
  0x24    4     layout        uint32: kRowMajor, kWave32Wmma16x16, kAie2RuntimeK
  0x28    4     ndim          uint32: число размерностей (1..4)
  0x2C    4     reserved_u32  uint32: 0 (выравнивание до 16 байт перед dims)
  0x30    16    dims[4]       uint32[4]: размерности тензора (неиспользуемые = 1)
  ```
  *Политика коллизий `name_hash`:* офлайн-упаковщик при записи `.ghalo` обязан проверять попарную уникальность `name_hash` всех тензоров модели и завершаться с ошибкой при совпадении хешей разных имён.
* **Единый упаковщик для GPU и NPU:** если образ собирается с `has_npu_tiles`, ведущие колонки плотных проекций prefill дополнительно сохраняются в block-float (`bfp16`) тайлинге `128x64x128` под `runtime_k` XCLBIN (как в коммитах `4f346915`, `1c00f953`, `35dd096c`), исключая перепаковку при старте.

---

### 3.2. Структуры буферов, KV-квантование и потоковое чтение снимков

Расширение [`QwenArenaMemoryLayout`](../../src/models/qwen/hip/detail/memory_layout.hpp) и [`QwenExecutionPolicy`](../../src/models/qwen/hip/execution_policy.hpp):

1. **Состояния 30 слоёв Gated DeltaNet (`recurrent_bytes`):**
   * Нерастущий буфер (`conv_state` в `FP32` + `ssm_state` в `FP32` или `BF16`), парный буфер отката + `replay_log` на 16 шагов (` AllocateSsmReplayLog` атомарен по всем 4 буферам с коммита `d81c883e`).
2. **Ускорение чтения памяти истории на длинном контексте (`GUFO_QWEN_SPARSE_ATTN` vs условный `Q8_0` KV-кэш):**
   * **Вариант А (проверяется первым на Этапе 2, 0 строк нового C++/HIP кода):** включение существующего `GUFO_QWEN_SPARSE_ATTN=1` ([`attention_sparse.hip`](../../src/models/qwen/hip/kernels/attention_sparse.hip), `QwenSparseDecodePartialsKernel<8>`). После порога `sparse_min_context = 8192` (дефолт старта прореживания — `290` блоков = `18 560` токенов) модель читает только наиболее важные блоки истории; на рабочем **`MTP1`** ориентир по архиву составляет **~+26 %** (`25,7 → 32,4 tok/s`; исторические `43,2 tok/s` относились к связке `AR + NPU`). Это приближение: требует проверки совместной работы с `MTP1` (`A1`) и оценки на прикладном наборе задач (`A2`, 50–100 примеров).
   * **Вариант Б (условный, по явному решению владельца на Этапе 3, только если `GUFO_QWEN_SPARSE_ATTN=1` не устроит по качеству, а телеметрия подтвердит частые истории >32K):** расширение `enum class QwenKvCacheStorage`:
     ```cpp
     enum class QwenKvCacheStorage : std::uint8_t {
       kFp32,
       kFp16,   // текущий production default
       kQ8_0,   // условный: 34 байта на 32 элемента (fp16 scale + 32 x int8)
       kTurbo6, // далёкий резерв для >262K после валидации Q8_0 (риск compute-bound на verify)
     };
     ```
   * **Граничные условия, уроки `tubo` (MIT) и идеи `strixite::kv8` (AGPL — только идеи) для `kQ8_0`:**
     * enum `QwenKvCacheStorage` дописывать **только в конец** с немедленным поднятием precision-bound в [`arena.cpp`](../../src/models/qwen/hip/arena.cpp#L252-L256); при смешанной точности использовать per-layer таблицу (`QwenKvLayerFormat`) и единую функцию расчёта смещения блока для writer и reader;
     * квантование `K` и `V` в `kQ8_0` выполняется на **GPU** в fused-эпилоге `QK-norm + RoPE + KV-write`;
     * чтение/деквантование `Q8_0` встраивается в wave64 split-K ядро [`QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>`](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip) (и split-K / prefill-пути, с опорой на FP64-тесты и ядра `Q6dense` MIT, но **без дубля `FP16`**; на prefill учитывать предупреждение `strixite::kv8` о `2,7×` замедлении при распаковке внутри WMMA-цикла — использовать staging; [`DequantizeQ8_0ToBf16Kernel`](../../src/models/qwen/hip/kernels/prefill_gemm.hip#L382) непригодно, так как выдаёт `BF16`, а внимание работает в `FP16`);
     * рассмотреть идею `strixite::kv8` с поканальной шкалой `K` (−42 % ошибки после RoPE) и точным несжатым хвостом `< 32` строк (`exact_from` для независимости от размера чанка prefill);
     * **явная несовместимость `kQ8_0` + `GUFO_QWEN_SPARSE_ATTN=1`:** оба режима одновременно не включаются (`QwenSparseDecodePartialsKernel` работает с `FP16` KV);
     * **требования к приёмке `kQ8_0`:** независимый CPU-оракул формата `Q8_0` KV, включение `QwenKvCacheStorage` в `Fingerprint()` и `numerics=...`, оценка по схеме «сравнительная таблица (`FP16` / `sparse` / `Q8_0` / `llama.cpp`) → независимое прочтение ответов на наборе из 50–100 задач → решение владельца», плюс не менее суток наблюдения на штатном `4 × 131072` до любых разговоров о `262144`.
3. **Потоковое чтение/запись снимков (`Snapshot Streaming` / `Restore`) и экономия памяти (`MEMTRIM`):**
   * Потоковая запись `StreamPersistent` (scratch 8 MiB) работает для рабочего Qwen+MTP; перенос на `DFlash2`/`auto` снят вместе с отказом от `DFlash2`.
   * Для восстановления с диска (`Restore`) — переход от полной аллокации `ReadImage` к потоковому чтению чанками 8–16 МиБ с двухфазной проверкой контрольной суммы до публикации в слот, с заимствованием zero-copy переноса из MIT `tubo` (`snapshot_copy.hip`, `hipHostGetDevicePointer` / `CopyAlignedSnapshot`) и переиспользуемого буфера (идея `strixite::HostBuffer`, плюс идея `strixite::MemoryGovernor` против гонок `EXP-N`).
   * Для разбора зазора `+4..7,5 ГиБ` к `llama.cpp` — рассмотреть инкрементальное выделение 2-МиБ KV-клеток по образцу MIT `tubo` (`GUFO_DEV_MEMTRIM_9`) вместо резервации всего буфера в конструкторе [`arena.cpp`](../../src/models/qwen/hip/arena.cpp).

#### Бюджет памяти на 128 ГБ UMA (4 сессии × 131 072 токена)
* **Веса (`UD-Q4_K_XL` или `Q6dense` `Q6_K` dense + `Q4_K` experts):** 22,0 – 23,5 ГБ (`~21,5 ГиБ` для `UD-Q4_K_XL` в памяти устройства, Пакет C).
* **MTP-head (`FP16` KV) + Vision (`BF16`):** ~1,8 ГБ + ~0,9 ГБ *(опциональные веса `DFlash2` `Q8_0` не загружаются в рабочем профиле `MTP1`)*.
* **DeltaNet state (30 слоёв, 4 сессии, с резервом отката):** ~1,6 ГБ.
* **KV-кэш Attention (10 слоёв × 2 KV-головы × `head_dim = 256`, профиль `4 × 128K = 524 288` токенов):**
  * сейчас в `FP16` (`20 480 Б/токен`): **10,00 ГиБ (10,74 ГБ)** (при `4 × 262K` — **20,00 ГиБ / 21,47 ГБ**; вся GPU-память с буферами — ~55,2 ГБ, см. [BASELINE.md](BASELINE.md));
  * условный `Q8_0` (`10 880 Б/токен`): **5,31 ГиБ (5,70 ГБ)** на `4 × 128K` — экономия **4,69 ГиБ (5,03 ГБ)** на 128K (и **9,38 ГиБ / 10,07 ГБ** на 262K) при почти двукратном снижении трафика чтения KV в decode.
* **Пер-поточные буферы prefill/MoE (4 сессии):** ~0,64 ГБ (`4 × 160 МиБ`).
* **RAM-кэш снимков продолжений:** 16 ГиБ в ОЗУ; **штатный дисковый кэш — 64 ГиБ** (мимо page cache; значение 16 ГиБ для диска использовалось только в стресс-тесте вытеснения `r3-earlyskip`, см. §2.10).

---

### 3.3. Пайплайн гибридного GPU + NPU префилла (условия безопасного включения)

По [NPU_MAP.md](NPU_MAP.md) код оффлоада (`QwenNpuFfn`, `runtime_k` XCLBIN для `K = 2048/2560/5120`, попарный lock-step `7093a45e`, pair-split `798d3c0f`) **уже есть в репозитории**.

#### Правила и архитектура включения (только после устранения платформенного зависания Strix Halo MC28 или на Gorgon Halo)
1. **Триггер разблокировки:** подтверждённое на Linux исправление Data Fabric (`ROCm/TheRock#8709`, `amd/xdna-driver#1605`, SMU ≥ 100.9.0) либо переход на плату без дефекта MC28. До этого `GUFO_ENABLE_NPU=OFF`.
2. **Конвейер:**
   * **Шаг 1 (Vision, если есть `<image>`):** выполняется на GPU до входа в языковой бэкбон; при текстовом запросе ViT не вызывается.
   * **Шаг 2 (Lock-Step двух банков `QwenPrefillBank`):** на чанках 1536–4096 токенов NPU считает ведущие колонки `qkv`, `gate`, `ssm_out` линейных слоёв (`1c00f953`) и `attn_q`, `attn_output` слоёв полного внимания (`35dd096c`), пока GPU параллельно считает внимание/DeltaNet и групповые GEMM 256 экспертов MoE.
   * **Шаг 3 (Защиты, обязательные к добавлению перед включением):**
     * перенос защиты `649824af` (проверка на non-finite logits после prefill, автоматическое отключение NPU и пересчёт на GPU при первой аномалии);
     * конечные таймауты на каждом `run.wait()` без фонового keepalive-потока;
     * включение SHA-256 `.xclbin` и потоков инструкций в численную идентичность дискового кэша (закрытие блокировки `6d52ac57` из находки **I3**).

---

### 3.4. Спецификация захваченных HIP-графов (`hipGraphExec_t`)

Расширение [`HipGraphDecodeExecutor`](../../src/core/hip/detail/hip_graph_decode_executor.hpp) и [`ResolveQwenGraphRejections`](../../src/models/qwen/hip/execution_policy.hpp) *(переписано по итогам аудита `T16`)*:

1. **Захват реальных горячих путей `MTP1` и разделение `C1`/`C4`:**
   * Сейчас единственный `instance_` в `HipGraphDecodeExecutor` используется только в [`ForwardToken`](../../src/models/qwen/hip/decode.cpp#L50-L108) (одиночный `AR`-декод `T = 1`) и **вообще не исполняется при `MTP1`**.
   * **Этап `T4` (пул графов + урок `TrimRollback` из MIT `Q6dense`):** хранение таблицы слотов `hipGraphExec_t` по ключу `(workload_identity, T, deep)` (где для рабочего `MTP1` `T = 2`, а бакеты/слоты могут покрывать и другие длины пакетов), а также обязательное уничтожение захваченных verifier-графов при уменьшении буферов отката или изменении `RecurrentIoIdentity` (`TrimRollback`, чтобы replay не обращался к освобождённой памяти).
   * **Этап `T5.1` (`C1`, одна сессия):** захват [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564) для `items.size() == 1` (`pos + row`) **и** захват шага черновика [`QwenMtpGpuExecutor::Run`](../../src/models/qwen/hip/mtp/executor.cpp#L232).
   * **Этап `T5.2` (`C4`, мультисеансы, `items.size() > 1`):** поддержка [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564) и [`ForwardTokenBatch`](../../src/models/qwen/hip/batched_decode.cpp#L236) с массивом per-session device-позиций и указателей арен сеансов (целевой фикс всплесков `27–39 мс` при `C4`). Захват на 2-м прогоне конфигурации с тихим fallback при ошибке без повторных попыток.
2. **Снятие отказа `kSplitKAttentionRequired` для обоих split-K ядер и порог `512`:**
   * Сейчас при длине истории выше `kSplitKDecodeAttentionMinContext` граф отключается (`ResolveQwenGraphRejections` выставляет `kSplitKAttentionRequired`). Кроме того, по [`LaunchSplitKAttentionHalfPartials`](../../src/models/qwen/hip/kernels/attention_decode.hip#L923-L956) при `deep = start_pos + 1 >= 512` запускается главное wave64-ядро 35B-A3B [`QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>`](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip#L24-L29), а при `128 <= seq_len < 512` — [`QwenDecodeSplitKAttentionHalfPartialsKernel`](../../src/models/qwen/hip/kernels/attention_decode.hip#L620-L627).
   * **Решение:** перевести скаляр `pos` на `const uint32_t*` в **обоих** ядрах (`attention_decode_verify_wave64.hip` и `attention_decode.hip`) и явно включать `deep = start_pos + 1 >= 512` в ключ графа (либо захватывать split-K графы только при `deep == true`), чтобы граф на `200` токенах не запекал более медленный 32-lane путь навсегда. Для передачи позиций **создать новый** pinned host + device-буфер `d_decode_step_params` по образцу `d_pos` в [`WriteKVCachePtrKernel`](../../src/models/qwen/hip/kernels/attention_decode_graph.hip#L14-L32). (`QwenDecodeSplitKAttentionReduceKernel` не принимает `pos`, а `SplitKDecodePartition` уже `__device__ __forceinline__` — их менять не нужно.)
3. **Предварительный прогрев `hipBLASLt` и MoE:**
   * На старте сервера до приёма первого HTTP-запроса принудительно прогревать планы `hipBLASLt` и `WarmMoeDecode`, чтобы убрать позднюю загрузку ELF-модуля (3,6 МиБ) и двухрежимную паузу 265/315 мс на первом запросе (находка этапа 3 третьего круга).

---

### 3.5. Адаптивная спекуляция под профиль Hermes Agent и OpenCode

По итогам Шага 1 ([DFLASH2_STEP1_REPORT_2026-11-17.md](DFLASH2_STEP1_REPORT_2026-11-17.md)) **`DFlash2` закрыт для `Qwen3.6-35B-A3B`** (`tool` медленнее `MTP1` на 8,76–16,13 % по времени хода и на 13,86 % по `decode tok/s`; Шаг 2 отменён). Ветка переключения `MTP1 (</think>) → DFlash2 (JSON/tools)` снята.

**Целевая задача `T7` (Cost Gate для `MTP1 ↔ AR` на `C2–C8`):**
* Вместо статического порога `GUFO_MTP_MOE_MAX_COHORT=4` ([`mtp/backend.cpp`](../../src/models/qwen/hip/mtp/backend.cpp#L265-L292)) при числе активных сеансов `active_slots >= 2` контроллер принимает решение «включать `MTP1` или делать `target-only` (`AR`) шаг» по измеренной стоимости пакета и реальному acceptance rate.
* Усиления при реализации `T7`: проверить порог отсечки неуверенных черновиков (`kMinDraftProbability = 0.5F` по образцу MIT `Q6dense`), рассмотреть чтение маржи top-2 (идея `strixite::MtpPick`) и калибровать пороги офлайн на собранной телеметрии `T2a` + собственной калибровке 35B (без копирования чужих констант `kMtpCycleMilliseconds`).

---

## 4. Единая дорожная карта (порядок выполнения)

Работы выполняются строго последовательно от закрытия долгов корректности к архитектурным ускорениям. Каждый стендовый шаг — только в согласованное с владельцем окно с сохранённым путём отката на **`b81812aa` / `be242d70`**.

### Этап 0. Закрытие эксплуатационного и тестового долга (`be242d70` / `004d8fa1`)
1. **Стендовый отсев DFlash2 против MTP1 на нагрузке Hermes Agent ([DFLASH2_EVAL_PLAN_2026-11-15.md](DFLASH2_EVAL_PLAN_2026-11-15.md)) — ЗАВЕРШЕНО (2026-11-17):**
   * Шаг 1 выполнен (`004d8fa1`, `C1`, `temperature = 0`, `hermes_synth` + `aux_title_request.json`, отчёт — [DFLASH2_STEP1_REPORT_2026-11-17.md](DFLASH2_STEP1_REPORT_2026-11-17.md)). `DFlash2` уступил `MTP1` на всех 5 случаях `tool` (−8,76 % в Проходе 1, −16,13 % в Проходе 2) и по полному эпизоду (−10,95 % / −13,00 %).
   * **Итог:** по критерию отказа **тема `DFlash2` для `35B-A3B` закрыта, Шаг 2 не нужен**.
2. **Доводка квалификации `be242d70` / `004d8fa1`:**
   * Синхронизация `ROADMAP_SPEC_01.md` (`T16`, ревизия 3 — 2026-11-17, ревизия 4 — 2026-11-18) и обновление [`docs/models/qwen3.6-35b-a3b/README.md`](../models/qwen3.6-35b-a3b/README.md) выполнены.
   * Часовой прогон памяти (`be242d70`, ядро `7.3.4`, PASS) и 4-пользовательский профиль (187,2 ток/с, PASS) выполнены 2026-11-18; недостижимая ветка `767c6018` в `StreamPersistentSnapshot` удалена (`4f934957`).
   * **Остаётся открытым:** независимое ревью коммитов варианта Б (`158711ad`, `004d8fa1`) и проверка реального автозапуска `gufo-faex2.service` при плановой перезагрузке машины.

### Этап 1. Телеметрия Hermes, набор оценок качества (50–100 примеров), зрение и API
1. **Пакет B (2-я очередь: `T1` `POST /tokenize`, `T2` `input_tokens`) и `T2a-1` — ЗАВЕРШЕНО И В ПРОДЕ (`be242d70`, 2026-11-17..18):**
   * Выполнены и проверены на живом стенде [`POST /tokenize`](../../docs/SERVER.md) (`6d793814`), OpenAI `input_tokens` (`4ab04ab8`), изоляция `ram_only` для `aux_title_request.json`, гистограммы/счётчики `/metrics` (`T2a-1`) и диагностика `cache_prefix_loss`. По мере работы `be242d70` под Hermes Agent накапливается реальное распределение длин историй в `/metrics`.
2. **Приоритеты планировщика и доводка кэш-политики (`T2a-2`) — ЗАВЕРШЕНО И В ПРОДЕ (`b81812aa`, launcher v17, 2026-11-18):**
   * В `faex2` реализованы и выведены в прод (`b81812aa`/`e5af3a93…`, откат на `be242d70`/`964e84e5…`): приоритет коротких запросов и `others_decoding`, одноразовая токенизация и удержание `system_prefix_tokens >= 512`, `fe2694bc` `IsBranchPoint`, динамический `MemoryGovernor` и `GrammarCache` LRU (`ToolParameters = 64`, `BindConstraint = 16`) — см. [дизайн `T2a-2`](../superpowers/specs/2026-11-18-t2a2-scheduler-cache-design.md).
   * Независимое fresh-context ревью: **`APPROVE`** (`45f0b8eb`, `b81812aa`). Подтверждено на стенде `faex2`: `check-pr` (`56/56`), `cpu-sanitizer` (`9/9`), Qwen/HIP/serve GPU (`77/77`, широкий `gpu-fast` без `#190`: `151/159`), `qwen_snapshot_streaming_test` (PASS), `qwen_layered_prefill_test` (PASS), `qwen_vision_serving_test --append-only` (PASS), живой `--mmproj` (`images` 4/4 PASS), matched pair vs `be242d70` + живой smoke (`7/7` PASS) и **часовой прогон памяти PASS** (RSS `+0,13 ГиБ`, GTT `−0,06 ГиБ`, `1560` запросов, ядро CLEAN; час не заменяет 24/7). Долг Этапа 1 по зрению без `--append-only` (`35/36 → 36/36`) остаётся открытым в п. 3 ниже.
3. **Прикладной набор оценок качества (50–100 примеров) и решение по контракту `Q1` / зрению:**
   * Собрать фиксированный набор из **50–100 прикладных задач семейного чата** (русские инструкции, школьные объяснения, таблицы/OCR, многошаговые инструменты Hermes), чтобы оценивать приближения (`sparse`, `Q8_0`, `Q6_K`) по реальной полезности ответов.
   * Согласовать с владельцем реалистичные допуски `Q1` (с учётом собственного шума арифметики `0,003–0,005`), устранить расхождение между холодным чтением изображения и продолжением из кэша (`model-vision-serving-mtp` 35/36 → **36/36 PASS**) и проверить опциональный даунскейл крупного текста (48–96 pt, `faex2_vision_probes.py`).

### Этап 2. Дешёвая проверка `GUFO_QWEN_SPARSE_ATTN=1`, пул HIP-графов и Cost Gate спекуляции
1. **Проверка готового разреженного внимания (`T15`, `GUFO_QWEN_SPARSE_ATTN=1`, [DECODE_DEPTH.md](DECODE_DEPTH.md) п. 3, 0 строк нового кода):**
   * Проверить семантику `QwenSparseDecodePartialsKernel<8>` с `MTP1` (`A1`: rollback/границы `8192` и `290` блоков), замерить скорость на 16K–128K (`A2`, ориентир в рабочем `MTP1` **~+26 %**, включая повторное сравнение с `llama.cpp` после `6bcf0c3b`) и оценить ответы на прикладном наборе 50–100 задач.
2. **Снятие `kSplitKAttentionRequired`, пул графов и профиль `MTP2/3` (`T3–T5b`):**
   * `T3`: device-`pos` (`d_decode_step_params` по образцу `d_pos` в `WriteKVCachePtrKernel`) для **обоих** split-K ядер (`attention_decode_verify_wave64.hip` с порогом `deep = start_pos + 1 >= 512` и `attention_decode.hip`).
   * `T4` + `T5a`: пул графов с учётом `deep` и `TrimRollback` (MIT `Q6dense`) + прогрев `hipBLASLt` и `WarmMoeDecode` на старте.
   * `T5`: раздельно `T5.1` (`C1`: `ForwardVerificationBatch` + `QwenMtpGpuExecutor`) и `T5.2` (`C4`: `ForwardVerificationBatch` + `ForwardTokenBatch` с per-session device-позициями).
   * `T5b`: профилирование просадки `MTP2/3` на 2–4 потоках (−5..−33 %) и проверка группировки `by_expert` на `down_exps` при `K = 512` (`Q4_K/Q5_K/Q6_K`).
3. **Адаптивный контроллер спекуляции (`T7`, Cost Gate Пакета E):**
   * Заменить жёсткий порог `GUFO_MTP_MOE_MAX_COHORT` на динамическую оценку стоимости **`MTP1 ↔ AR` на `C2–C8`** (учитывая отсечку `0.5` из `Q6dense` и идею `MtpPick`/`capture` из `strixite`). Переключение на `DFlash2` и `StreamPersistent` для `DFlash2` сняты.

### Этап 3 (Условный, по решению владельца). Блочный KV-кэш `Q8_0` и потоковое чтение снимков (`Restore`)
1. **Потоковое восстановление снимков с диска (`Restore`) и разбор зазора памяти (+4..7,5 ГиБ к `llama.cpp`, `T12a`):**
   * Убрать чтение полного файла снимка (2,7+ ГиБ) в один буфер RAM при `Restore` (zero-copy перенос `snapshot_copy.hip` из MIT `tubo` + идеи `strixite::HostBuffer`/`MemoryGovernor`), изучить `GUFO_DEV_MEMTRIM_9` (инкрементальные 2-МиБ клетки вместо резервации всех `10,00 ГиБ` сразу).
2. **Внедрение `QwenKvCacheStorage::kQ8_0` (`T8–T12`, начинать только если телеметрия подтвердила частые >32K, а `GUFO_QWEN_SPARSE_ATTN=1` на Этапе 2 отклонён по качеству):**
   * Ожидаемая экономия на `4 × 131 072`: **4,69 ГиБ (5,03 ГБ)** (`10,00 → 5,31 ГиБ`). Независимый CPU-оракул, `enum` в конец + precision-bound + per-layer таблица (`tubo`), запрет `Q8_0 + sparse`, квантование `K`/`V` на GPU + чтение `Q8_0` в `QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>` (опора на MIT `Q6dense` без дубля `FP16`, staging на prefill по уроку `strixite::kv8`).
   * Приёмка по процедуре: сравнительная таблица (`FP16` / `sparse` / `Q8_0` / `llama.cpp`) → слепое/независимое прочтение ответов на наборе 50–100 задач → решение владельца → сутки наблюдения на `4 × 131072`.

### Этап 4. Смешанное квантование `Q6_K` dense + `UD-Q4_K_XL` experts (`T14`/`T14b`; `.ghalo` отложено)
1. **Эксперимент точности и скорости `Q6_K` dense/shared + `FP32/BF16` router + `UD-Q4_K_XL` experts (`T14` GGUF-рецепт + опционально `T14b` MIT-ядра `Q6dense`):**
   * После проверки лицензий HF и `q6dense.types` сравнить `ΔNLL`/KL (по методике `Q1`), `hermes_synth`, 50–100 задач и скорость prefill/decode против текущего `UD-Q4_K_XL`.
2. **~~Кэшируемая перепаковка `.ghalo`~~ (ОТЛОЖЕНО по итогам `T16`):**
   * Не делать до появления профиля, доказывающего нехватку перепаковки при загрузке в память.

### Этап 5. Гибридный GPU + NPU Prefill (только при закрытии платформенного бага Strix Halo)
1. При появлении стабильной прошивки без зависания MC28 (или на новом железе Gorgon Halo) — добавить автоотключение при non-finite logits (`649824af`), включить fingerprint `.xclbin` в идентичность кэша и провести безопасный ретест по чек-листу [NPU_MAP.md](NPU_MAP.md) §7.

---

## 5. Критерии приёмки (KPI)

Все критерии фиксируются **до** запуска стендовых окон; пороги не расширяются задним числом:

| Категория | Метрика / Проверка | Целевой порог приёмки |
| --- | --- | --- |
| **Корректность и тесты** | Набор GPU/CPU/model проверок (`r3-tests`) | **36 из 36 PASS** (включая `model-vision-serving-mtp` после закрытия Этапа 1); 0 ошибок в журнале ядра (`dmesg`/`journalctl`). |
| **Численное и прикладное качество** | `ΔNLL` / Perplexity, контракт `Q1` и прикладной набор (50–100 задач) | При оптимизациях без смены формата весов — побитовое совпадение либо `Δppl < 0.03 %`; для приближённого prefill — ворота `Q1`; для приближений контекста/весов (`GUFO_QWEN_SPARSE_ATTN`, `Q8_0` KV, `Q6_K`) — CPU-оракул + слепая сверка ответов на наборе из 50–100 семейных/Hermes задач и явное решение владельца. |
| **Основной клиент (Hermes Agent)** | Набор `hermes_synth` (15 случаев + `aux_title_request.json`) | 100 % валидных вызовов инструментов и JSON-схем; критерий принятия `DFlash2` (≥ 15 % на `tool` / эпизоде) проверен на Шаге 1 (`DFlash2` закрыт, рабочим остаётся `MTP1`). |
| **Второй клиент (OpenCode 2.3.0)** | Изолированный `opencode_agent.py` (`tools`, `edit`, `verify`, `mcp`) | **4 из 4 PASS**, `ELIGIBLE`, 100 % настоящих цепочек `вызов → результат → следующий запрос`, 0 утечек служебной разметки. |
| **Скорость (Prefill / Decode / TTFT)** | Парное сравнение `контроль ↔ кандидат` (2+2 или 4+4 с чередованием) | В каждой затронутой ячейке/фазе: не хуже контроля более чем на **5 % и 3 мс** одновременно (без усреднения запросов). На длинной истории (64K–128K) от `Q8_0` KV и HIP-графов — измеримый прирост `decode tok/s` на C1 и C4. |
| **Память и 24/7 стабильность** | Часовой прогон `compose --duration 3600` (отсчёт раз в 1 с) | Рост максимума и медианы **RSS, GTT, VRAM и GTT+VRAM ≤ 1 ГиБ** между первыми и последними 20 минутами; 0 потерь активных сеансов при отменах (`cancel`); проверенный возврат на запасной бинарник без пересборки за < 60 с. |
