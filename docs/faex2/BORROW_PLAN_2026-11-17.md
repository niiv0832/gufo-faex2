# BORROW_PLAN_2026-11-17: План работ по заимствованиям и архитектурным идеям

Дата: 2026-11-17 (обновлено по итогам двойного аудита)  
Статус: **уточнённый проект для согласования владельцем; не разрешение на реализацию или запуск испытаний**  
Основание: [BORROW_EVAL_2026-11-17.md](BORROW_EVAL_2026-11-17.md) (ревизия по результатам аудита) + [ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md) (`90f73f1a`, `26904442`) + `NEXT_STAGE_SPEC_2026-11-14` (§1.2, §7, §8, §10 — сохранённые решения владельца).

## 0. Правила выполнения (действуют для всех задач)

1. Порядок работ задаёт владелец и [ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md); ни одна задача ниже не стартует без явного поручения. Исторические планы — не команды на исполнение.
2. Каждая задача — отдельная ветка логики: небольшие Conventional Commit'ы только своих путей, noreply-идентичность, обновление [CONTEXT.md](CONTEXT.md) и [CHANGELOG.md](CHANGELOG.md).
3. Численные изменения не смешивать друг с другом и со структурными правками: один кандидат — один вопрос, matched-контроль по [TESTING.md](../../docs/TESTING.md) (порог 5 % и 3 мс, без усреднения).
4. Стенд: только в согласованное окно, с возвратом на рабочую роль (`004d8fa1` / актуальную) и сохранением raw. Долгие операции — устойчивы к обрыву SSH.
5. **Правовые правила (строго!):**
   * **`strixite` (GNU AGPL-3.0):** запрещено переносить **ни одной строки кода** (наш `gufo serve` — сетевой сервис). Можно брать **только инженерные идеи** (помечены `[идея strixite]`), реализуя их с нуля под нашу архитектуру.
   * **`tubo` и `Q6dense` (MIT):** код переносим при сохранении copyright-нотисов, записей в `NOTICE`/`THIRD_PARTY_NOTICES.md` и обновлении `VENDOR.md` при правках `mmq`.
   * **Веса HuggingFace:** проверять лицензии карточек до скачивания на стенд.

**Словарь сокращений:** C1/C4 — 1 и 4 одновременных пользователя; «Q1-ворота» — контракт точности приближённого prefill ([PREFILL_QUALITY_CONTRACT_DRAFT.md](PREFILL_QUALITY_CONTRACT_DRAFT.md)); `hermes_synth` — набор 15 запросов Hermes Agent + `aux_title_request.json` ([tests/functional/fixtures/hermes_synth](../../tests/functional/fixtures/hermes_synth)); `A1/A2` — проверка семантики и качества/скорости sparse-attention.

---

## Блок A. Профиль Hermes Agent: API, телеметрия, планировщик и кэш (Этап 1; вердикт «ДЕЛАТЬ»)

Закрывает все **четыре** подзадачи [ROADMAP_SPEC_01.md §1.2 п.6](ROADMAP_SPEC_01.md#L41) и Этапа 1 п.1–2,4.

### T1. `/tokenize` из апстрима `6d793814` (~60–80 строк)
* **Файлы:** [http_server.cpp](../../src/cli/serve/http_server.cpp) (`LlamaTokenize` + `register_routes`:1851–1870 рядом с `AnthropicCountTokens`:1328), [inference_backend.cpp](../../src/cli/serve/inference_backend.cpp#L5858) (`runner().Tokenize(...)`), [text_generation_backend.hpp](../../src/cli/serve/text_generation_backend.hpp#L383-L386).
* **Шаги:** ручной перенос (3-way патч конфликтует) → маршрут `POST /tokenize` → тест в `tests/cli/http_server_test.cpp`.
* **Приёмка:** юнит-тест, `check-format.py`, HTTP smoke на сервере (совпадение с `TokenizeControl`).

### T2. Input token counts из апстрима `4ab04ab8` (~180–250 строк, после T1)
* **Файлы:** `http_server.cpp` (аккуратный рефакторинг вокруг `OpenAiResponses`:1127; маршруты `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`), `openai_chat.cpp/.hpp` (`CountOpenAiInputTokens` поверх `count_chat_tokens`), функциональный тест `tests/functional/openai_count_tokens.py`.
* **Приёмка:** перенос теста (локальный мок + стенд), сверка пары запросов `OpenAiResponses` до/после.

### T2a. Телеметрия длин, планировщик и защита кэша под Hermes Agent (`[идеи strixite]` + апстрим)
* **Телеметрия (Этап 1 п.1):** `[идея strixite::capture]` опциональный бинарный/лёгкий лог на завершённый запрос (`prompt_n`, `tokens_n`, `tools`, `one_shot`, пошагово `{drafted, accepted, emitted}`) — собирает гистограмму реальных длин историй Hermes (предусловие для `T15 sparse` и `Q8_0`) и базу для офлайн-прогона cost gate.
* **Приоритет очереди (Этап 1 п.2):** собственная доработка в [text_generation_scheduler.cpp](../../src/cli/serve/text_generation_scheduler.cpp) — приоритет коротких семейных запросов перед стартом/очередью длинного агентского prefill + `[идея strixite::GpuArbiter::others_decoding]` уменьшение чанка prefill при наличии активных декодеров.
* **Защита кэша и диагностика потерь префикса (Этап 1 п.2):**
  * проверить применимость апстрима **`fe2694bc`** (#1104) и **`8ab834a5`** (#903) к нашему `ContinuationCache` / планировщику (`68e8475d` и `93af45d4` уже есть в `HEAD`);
  * `[идея strixite::prompt_cache]` пометка одноразовых запросов без `tools` (`aux_title_request.json`) как **`ram_only`** (кэшируется в RAM, **не пишется на диск** и не вытесняет длинные диалоги); чёткое удержание общего префикса `System` (~55K символов) при сжатии истории и ограничение фонового сброса (`r3-earlyskip`);
  * `[идея strixite::resume_loss]` логирование при потере >512 токенов общего префикса с указанием номера сообщения, где оборвалось совпадение в DeltaNet;
  * `[идея strixite::GrammarCache]` LRU-кэш скомпилированных JSON-схем (`aux_title_request.json`, до ~16 записей) и опционально `[идея strixite::MemoryGovernor]` (единое чтение `/proc/meminfo` в цикле вытеснения против гонок `EXP-N`).

---

## Блок F (приоритетный на Этапе 2!). Проверка готового `GUFO_QWEN_SPARSE_ATTN=1` (`T15`)

По [ROADMAP_SPEC_01.md §3.2 п.2 и §4 Этап 2 п.1](ROADMAP_SPEC_01.md#L254-L255) эта задача идёт **строго ДО** разработки `Q8_0` KV (Блока D). Код уже в дереве (`34e61a15`, [attention_sparse.hip](../../src/models/qwen/hip/kernels/attention_sparse.hip)).

### T15. A1 — семантика, затем A2 — качество/скорость (0 строк нового кода)
* **Предусловие:** телеметрия из `T2a` (оценка доли реальных длинных историй Hermes).
* **A1 (дёшево, до широких замеров):** операторный тест — логиты первой verify-строки не зависят от содержимого draft-строк; `accept/reject/rollback/EOS/tool tokens` на фиксированных токенах; границы `tile 32 / key block 64 / min_context 8192` и точка старта прореживания `290` блоков (`18 560` токенов); sparse-идентичность снапшотов.
* **A2 (после A1):** сравнение `GS` vs `G16` (`FP16-dense`) + сравнение с закреплённым `llama.cpp` после `6bcf0c3b` на прикладном наборе (50–100 задач, русские пересказы, `tools`, картинки на длинной истории); замер отдельно AR и `MTP1` (ориентир по архиву — **~+26 %** на `MTP1`: `25,7 → 32,4 tok/s`), `C1` и `C4`.
* **Итог:** решение владельца (принять профиль / ограничить / отклонить по качеству — только при отклонении открывается условный `Q8_0` в Блоке D).

---

## Блок B. Бакетные HIP-графы и стартовый прогрев (Этап 2; вердикт «ДЕЛАТЬ», ~600–1200 строк)

Переписано с исправлением технических ошибок исходного плана. Источники: [gufo-laguna](../../src/models/laguna/gpu/engine.cpp) и **`Q6dense` (MIT, [executor.cpp](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/executor.cpp))**. *(В `strixite` HIP-графов нет!)*

### T3. Device-pos для обоих split-K ядер + порог `512` (~180–300 строк)
* **Файлы:**
  * [attention_decode_verify_wave64.hip](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip#L24-L29) — **главное ядро 35B-A3B при `seq_len >= 512`** (`QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>`): перевести скаляр `pos` на `const uint32_t* d_pos` (либо `d_decode_step_params`);
  * [attention_decode.hip](../../src/models/qwen/hip/kernels/attention_decode.hip#L620-L627) (`QwenDecodeSplitKAttentionHalfPartialsKernel`, используется при `128 <= seq_len < 512`) — аналогичный device-`pos`;
  * `LaunchSplitKAttentionHalfPartials` ([attention_decode.hip:923](../../src/models/qwen/hip/kernels/attention_decode.hip#L923)) — убрать тихое запекание порога `deep = start_pos + 1 >= 512` без отражения в ключе графа (разделять ключ графа по `deep` либо, по образцу `laguna`, захватывать split-K графы только при `>= 512`);
  * [decode_step.cpp:180-191](../../src/models/qwen/hip/decode_step.cpp#L180-L191), [batched_decode.cpp](../../src/models/qwen/hip/batched_decode.cpp), `executor.hpp` / `arena.cpp` — создать pinned host + device буфер `d_decode_step_params` по аналогии с `d_pos` в [`WriteKVCachePtrKernel`](../../src/models/qwen/hip/kernels/attention_decode_graph.hip#L14-L32).
* **Исключено как ошибочное из прежнего плана:** `QwenDecodeSplitKAttentionReduceKernel` ([attention_decode.hip:836-839](../../src/models/qwen/hip/kernels/attention_decode.hip#L836-L839)) **не имеет** параметра `pos`, а `split_count` всегда `32` — менять не нужно; `SplitKDecodePartition` ([attention_decode_split_k.hpp:49](../../src/models/qwen/hip/kernels/attention_decode_split_k.hpp#L49)) **уже `__device__ __forceinline__`**.
* **Приёмка:** побитовая идентичность до/после; снятие отказа `kSplitKAttentionRequired`; тест графа через порог `512`; замер `tg 64K` без регрессии >5 %/3 мс. Sparse-путь в граф не брать.

### T4. Пул графов по `(workload_identity, T, deep)` + урок `TrimRollback` (~120–200 строк, после T3)
* **Файлы:** [hip_graph_decode_executor.hpp](../../src/core/hip/detail/hip_graph_decode_executor.hpp#L141-L147) (массив слотов `exec` + per-slot `capture_attempted`/`failed` вместо единичного `instance_`), [decode.cpp](../../src/models/qwen/hip/decode.cpp), [arena.cpp](../../src/models/qwen/hip/arena.cpp).
* **Урок `Q6dense` (`TrimRollback`, [executor.cpp:192-206](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/executor.cpp#L192-L206)):** при освобождении/сокращении буферов отката или изменении `RecurrentIoIdentity` обязательно уничтожать захваченные verifier-графы, запекшие старые указатели!

### T5. Захват горячих путей `MTP`/`DFlash2` и многосеансового `C4` (~300–650 строк, после T4)
* **Разделение на 2 подэтапа (устранение ошибки смешения C1 и C4):**
  1. **`T5.1` (Одиночная сессия C1):** захват [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564) для `items.size() == 1` (`T = 1..8`, `pos + row`) **плюс** захват шага черновика [`QwenMtpGpuExecutor::Run`](../../src/models/qwen/hip/mtp/executor.cpp#L232) по образцу `Q6dense` (без них при включённом `MTP1` графы не работают!).
  2. **`T5.2` (Многосеансовые батчи C4, `items.size() > 1`):** поддержка [`ForwardVerificationBatch`](../../src/models/qwen/hip/batched_decode.cpp#L564) и [`ForwardTokenBatch`](../../src/models/qwen/hip/batched_decode.cpp#L236) с массивом per-session device-позиций/смещений для разных контекстов и арен сеансов (прямой удар по всплескам 27–39 мс при `C4`). Захват на 2-м прогоне конфигурации, тихий fallback при ошибке без ретраев. *(Замечание: `kMoeNotCapturable` в `decode.cpp:63` по умолчанию уже `false` — не является блокером.)*
* **Приёмка:** бит-в-бит совпадение captured/uncaptured для `T=1,2,4,6,8` на `C1` и `C4`; проверка утечек графов; замер снижения пауз `C4`.

### T5a. Предварительный прогрев `hipBLASLt` и `WarmMoeDecode` на старте (ROADMAP §3.4 п.3, ~30–60 строк)
* Принудительно прогревать планы `hipBLASLt` (ELF 3,6 МиБ) и `WarmMoeDecode` при старте сервера до приёма 1-го HTTP-запроса — закрытие паузы `265/315 мс` на первом запросе.

### T5b. Диагностика просадки `MTP2/3` (§2.Б п.7) и проверка grouped `down_exps` (Этап 2)
* Снять профиль на 2–4 потоках с `MTP2/MTP3` (−5..−33 %): проверить вклад отсутствия графов на `MTP`, собственного пути внимания внутри MTP-головы и отсутствия группировки `by_expert` в [`qfn_mmq.hip.cpp:505-508`](../../src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp#L505-L508) (`K=512` и `Q4_K/Q5_K/Q6_K` на 35B против условия `K==640 && (Q5_1||Q8_0)`). Если профиль подтвердит заметную долю `down_exps` — точечно перенести grouped expert-down для `Q4_K/Q5_K/Q6_K` из MIT `Q6dense` (`EXPERIMENTS.md` #21).

---

## Блок C. Cost gate спекуляции (вердикт «ДЕЛАТЬ по частям»)

### T6. Апстрим `1167a944` в `qwen38_flash_next` (~20–30 строк; 27B-долг, низкий приоритет)
* **Файлы:** [`src/models/qwen38_flash_next/mtp_policy.hpp`](../../src/models/qwen38_flash_next/mtp_policy.hpp) (`MtpPolicyConcurrency`), `engine.cpp`, `sampling.hpp:44`. Не влияет на целевой 35B-A3B, но дёшево убирает расхождение с апстримом (`2caf2a82` уже учтён у нас).

### T7. 35B Cost Gate вместо `GUFO_MTP_MOE_MAX_COHORT=4` (~150–250 строк; **строго после Этапа 0**)
* **Файлы:** [`mtp/backend.cpp:265-292`](../../src/models/qwen/hip/mtp/backend.cpp#L265-L292), [`mtp_policy.hpp:85-90`](../../src/models/qwen/mtp_policy.hpp#L85-L90), [`auto_draft.cpp:183-263`](../../src/models/qwen/hip/auto_draft.cpp#L183-L263), таблицы [`dflash_policy.hpp:321-360`](../../src/models/qwen/dflash_policy.hpp#L321-L360).
* **Два сценария (оба учтены):**
  * если `DFlash2` отклонён на Этапе 0 — cost gate управляет включением/выключением **`MTP1 ↔ AR` на C2–C8** вместо жёсткого `cohort <= 4`;
  * если `DFlash2` прошёл порог 15 % — добавляется переключение `MTP1 (</think>) → DFlash2 (JSON/tools)` + `StreamPersistent` (≤16 МиБ) для DFlash2.
* **Усиление заимствованием и идеями:**
  * проверить порог отсечки неуверенных черновиков (`kMinDraftProbability = 0.5F` из MIT `Q6dense`, `EXPERIMENTS.md` #10 — +1 % скорости без изменения greedy);
  * `[идея strixite::MtpPick]` при необходимости читать `best_v - second_v` в одном 64-байтном копировании;
  * `[идея strixite::capture]` калибровать пороги офлайн на собранных логах `T2a` + собственной калибровке 35B (чужие `kMtpCycleMilliseconds` не копировать!).

---

## Блок D. Потоковое чтение `Restore`, разбор памяти и **УСЛОВНЫЙ** `Q8_0` KV (Этап 3)

### T12a. Потоковый `Restore` и экономия базовой памяти KV (§2.В п.9–10, безусловно в Этапе 3)
* **Потоковое чтение снимков `Restore`:** заменить аллокацию всего многогигабайтного файла в RAM при чтении чанками 8–16 МиБ с двухфазной проверкой checksum. Заимствовать из **MIT `tubo`** [`snapshot_copy.hip`](../../../_borrow-eval/tubo/src/core/hip/snapshot_copy.hip) (zero-copy перенос через `hipHostGetDevicePointer` / `CopyAlignedSnapshot`) + `[идея strixite::HostBuffer]` (переиспользуемый `mmap`/`mremap` буфер без повторных page fault).
* **Разбор зазора `+4..7,5 ГиБ` к `llama.cpp`:** изучить в **MIT `tubo`** **`GUFO_DEV_MEMTRIM_9`** (инкрементальное выделение 2-МиБ KV-клеток вместо резервации всех `10,00 ГиБ` сразу в конструкторе `arena.cpp` на 4 сессии × 128K) и `MEMTRIM_4/7`, а также `[идея strixite::MemoryPlan]` (динамический бюджет старта).

### T8–T12. `Q8_0` KV-кэш (~300–600+ строк) — **СТРОГО УСЛОВНО!**
> **Условие старта (по `ROADMAP_SPEC_01` §4 Этап 3 п.2):** начинать **ТОЛЬКО** по явному решению владельца, если: (а) телеметрия `T2a` подтвердила частые длинные истории >32K **и** (б) проверка `T15 sparse` на Этапе 2 отклонена по качеству. Ожидаемая экономия памяти на `4×131 072`: **4,69 ГиБ (5,03 ГБ)** (`10,00 → 5,31 ГиБ`).

* **Архитектурные уроки `tubo` (MIT) и идеи `strixite::kv8`:**
  * enum `QwenKvCacheStorage` **дописывать только в конец** и поднять precision-bound в [`arena.cpp:252-256`](../../src/models/qwen/hip/arena.cpp#L252-L256); при смешанной точности — таблица по плоскостям (`QwenKvLayerFormat`); одна функция смещения блока (`QwenKvHeadBlockOffset`) на writer+reader;
  * `[идея strixite::kv8]` рассмотреть поканальную шкалу для `K` (−42 % ошибки после RoPE) vs блочной 32; держать неполный хвост `<32` в `FP16` (`exact_from` инвариант — независимость от чанков prefill!).
* **Шаги `T8–T12`:**
  * **T8:** спецификация формата + автономный CPU-оракул (0 строк движка).
  * **T9:** квантование `K/V` в эпилоге `QK-norm + RoPE + KV-write` на всех writers.
  * **T10:** чтение/деквантование в **`QwenVerifySplitKAttentionHalfPartialsWave64Kernel<8>`** ([attention_decode_verify_wave64.hip](../../src/models/qwen/hip/kernels/attention_decode_verify_wave64.hip)) и split-K / prefill-путях (с опорой на готовый код чтения и FP64-тесты 35B-A3B из **MIT `Q6dense`**: [`QuantizeKvKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L1704) + [`GqaDecodeAttentionWmmaKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L1744), но с **однослойным** хранением `Q8_0` без дубля F16! На prefill помнить предупреждение `strixite::kv8` о `2,7×` замедлении при распаковке внутри WMMA-цикла — использовать staging). **Не опираться на [`DequantizeQ8_0ToBf16Kernel`](../../src/models/qwen/hip/kernels/prefill_gemm.hip#L382): оно выдаёт `BF16`, а внимание использует `FP16`!**
  * **T11:** включение `QwenKvCacheStorage` в `Fingerprint()` / `numerics=...` кэша, промах на старые снапшоты.
  * **T12:** приёмка (AR/verify/rollback/графы/vision/persistence + Q1 + 50–100 задач + сутки на `4×131072`; `sparse + Q8_0` строго не включать вместе).

### T13. `kTurbo6` (~1000+ строк) — **Далёкий резерв (низкий приоритет)**
* Только после `T12` и при реальной нехватке `4×262K`. Учитывать риск из `tubo` `BENCHMARKS.md` `N32`: 8-row verify-внимание на `turbo6` упиралось в вычисления/LDS (`68,4` / `49 ГБ/с`), что может свести на нет выигрыш для нашего `MTP1`.

---

## Блок E. Эксперимент `Q6dense` и `.ghalo` (Этап 4; вердикт «ПОПРОБОВАТЬ»)

### T14. GGUF-рецепт `Q6dense` + приёмка (0 строк кода, 1 вечер стенда)
1. Проверить лицензии на HuggingFace карточек `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` и `ligamentexceed/...-Q6dense-GGUF`.
2. Проверить `q6dense.types`: убедиться, что `kNorm` / `kSsmParameter` остаются `F32`, а роутер совместим с `SupportsMoeRouterType` ([weights.cpp:62-95](../../src/models/qwen/weights.cpp#L62-L95)).
3. Сквантовать из `unsloth BF16` через `llama-quantize --imatrix ... --tensor-type-file q6dense.types ... Q4_K_M 16`.
4. Прогнать `Q1` + `hermes_synth` + 50–100 задач + matched-скорость против `UD-Q4_K_XL`. `.ghalo` не делать.

### T14b. Оценка MIT-ядер `Q6dense` под 35B-A3B (после T14, опционально)
* Если профиль подтвердит потенциал скорости 64K–131K: оценить точечный перенос под MIT-нотисами `DenseQ6KVecKernel` (перепаковка `Q6_K` dense при загрузке без дубликатов `PrefillQ8`), `SmallGemmSplitF16Kernel` для роутера и проверку схемы нормализации [`MtpHiddenKernel`](../../../_borrow-eval/gufo-Qwen3.6-35B-A3B-Q6dense/src/models/qwen36_35b_a3b/kernels/rocm/kernels.hip.cpp#L2176) против [`mtp/executor.cpp`](../../src/models/qwen/hip/mtp/executor.cpp#L259-L322) на acceptance rate.

---

## Блок G. Синхронизация `ROADMAP_SPEC_01` (`T16`, 0 строк кода)

### T16. Какие правки требуются в [ROADMAP_SPEC_01.md](ROADMAP_SPEC_01.md) (уточнено по аудиту)
1. **§1.2 п.1 (`Бакетные HIP-графы`):** убрать ошибочно указанную `strixite` (графов 0), добавить **`Q6dense` (MIT)**; в **§3.4** поправить описание на реальные ядра (`attention_decode_verify_wave64.hip` + порог `512` + `ForwardVerificationBatch` + `MTP` + разделение `C1`/`C4` + `TrimRollback`) и исправить устаревшее описание `d_decode_step_params (аналогично attention_decode_graph.hip)` на «создать буфер по образцу `d_pos` в `WriteKVCachePtrKernel`».
2. **§1.2 п.2 (`MoE`):** зафиксировать, что grouped/indirect `gate/up` уже реализованы (и ссылку на переименование `GetMoeFfnScratch` → `QwenMoeScratch`), но **оставить открытыми** диагностику `MTP2/3` (§2.Б п.7) и проверку `by_expert` на `down_exps` при `K=512`.
3. **§1.2 п.3 (`Cost gate`):** уточнить, что в `gufo-laguna` гейта нет; сослаться на `Q6dense` (`0.5` фильтр) и идеи `strixite` (`capture`, `MtpPick`).
4. **§1.2 п.5 и §3.1 (`Q6dense`, `.ghalo`):** пометить `.ghalo` отложенным; отметить MIT-код `Q6dense` (`qwen36_35b_a3b`) и идеи `strixw`/`WMMA` из `strixite` (AGPL — только идеи).
5. **§1.2 п.6 (`Hermes`), §3.2 (`Q8_0` / `Restore` / `MEMTRIM`):** дополнить идеями `strixite` (`ram_only`, `System/Checkpoint`, `resume_loss`, `GrammarCache`, `MemoryGovernor`, `HostBuffer`, `kv8` поканальный K и предупреждение по prefill) + уроками `tubo` (`enum`-конец, per-layer таблица, `snapshot_copy.hip`, `GUFO_DEV_MEMTRIM_9`) + выровнять `ГиБ`/`GB` (`10,00 ГиБ → 5,31 ГиБ`, экономия `4,69 ГиБ / 5,03 ГБ`).
6. *(Исправлено по аудиту)* **Не добавлять** «новый этап Sparse-валидация» — этап `GUFO_QWEN_SPARSE_ATTN=1` уже есть в `90f73f1a` (§4 Этап 2 п.1); лишь уточнить ожидаемый прирост для `MTP1` (~+26 %).

---

## 1. Сводный порядок и трудозатраты (выровнено с `ROADMAP_SPEC_01`)

| Приоритет | Этап | Задача | Объём | Зависимости | Стенд |
| :---: | :---: | :--- | :--- | :--- | :--- |
| 1 | Док. | **T16** синхронизация `ROADMAP_SPEC_01` | 1 час | план утверждён | нет |
| 2 | Этап 1 | **T1** `/tokenize` (`6d793814`) | 0,5–1 день | — | нет (мок) + smoke |
| 3 | Этап 1 | **T2** input token counts (`4ab04ab8`) | 1–2 дня | T1 | нет + окно Этапа 1 |
| 4 | Этап 1 | **T2a** телеметрия `capture`-идея + приоритет планировщика + кэш (`ram_only`/`fe2694bc`/`resume_loss`/`GrammarCache`) | 3–5 дней суммарно | — | окна Этапа 1 |
| 5 | **Этап 2** | **T15 sparse A1/A2** (`GUFO_QWEN_SPARSE_ATTN=1`, 0 строк кода!) | A1: 1–2 дня; A2: 2–3 окна | телеметрия `T2a` | **первым на Этапе 2** |
| 6 | Этап 2 | **T3** device-pos для `wave64` + `split-K` + порог `512` | 2–4 дня | — | окно Этапа 2 |
| 7 | Этап 2 | **T4** пул графов + `TrimRollback` + **T5a** прогрев `hipBLASLt` | 2–3 дня | T3 | окно Этапа 2 |
| 8 | Этап 2 | **T5** verify + MTP + `C4` захват (`T5.1`, `T5.2`) | 4–7 дней | T3, T4 | окна Этапа 2 |
| 9 | Этап 2 | **T5b** профиль `MTP2/3` + проверка grouped `down_exps` (`K=512`) | 1–3 дня | — | окна Этапа 2 |
| 10 | Этап 2 | **T7** 35B cost gate (+ фильтр `0.5` `Q6dense`) | 3–5 дней | Этап 0 + данные `T2a` | окно Этапа 2 |
| 11 | Долг 27B | **T6** `1167a944` в `qwen38_flash_next` | 0,5 дня | — | нет (юнит) |
| 12 | Этап 3 | **T12a** потоковый `Restore` (`snapshot_copy.hip`) + разбор `+4..7,5 ГиБ` (`MEMTRIM_9`) | 3–5 дней | — | окна Этапа 3 |
| 13 | Этап 3 | **T8–T12 `Q8_0` KV** (**строго условно!**) | 2–3 недели | **только при провале T15** + >32K в T2a | окна Этапа 3 + сутки |
| 14 | Этап 4 | **T14** GGUF-эксперимент `Q6dense` (+ опц. **T14b** MIT-ядра 35B) | T14: 1 вечер | лицензии HF + `q6dense.types` | окно Этапа 4 |
| 15 | Резерв | **T13** `kTurbo6` (далёкий резерв) | 2–3+ недели | T12 + решение при нужде >262K | окна |

Оценки сроков/строк ориентировочные (±50 %), без учёта очередей на стенд. Этап 0 roadmap (`DFlash2`-отсев и закрытие долга `004d8fa1`) остаётся первым до любых крупных доработок движка.
