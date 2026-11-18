# T2a-2: приоритеты планировщика, доводка кэша и закрытие хвостов

Дата: 2026-11-18. Статус: **проект дизайна на согласование владельцем, реализация не начата**.
Исходная ревизия: `ad102a3c`, ветка `faex2` (текущий продакшен на стенде: `be242d70`, резерв: `004d8fa1`).

Основание: [план работ `T2a`](../../faex2/BORROW_PLAN_2026-11-17.md),
[оценка заимствований, §7 и §9–10](../../faex2/BORROW_EVAL_2026-11-17.md),
[ROADMAP, Этап 0–1](../../faex2/ROADMAP_SPEC_01.md).

---

## 1. Итоги аудита кода: что уже закрыто, что нужно добавить, что нецелесообразно

### 1.1. Приоритет коротких запросов и уступчивый prefill при активных декодерах

- **Что уже работает (`21736dd0`, `afcff8fb`):**
  - `QwenTextRunner::Descriptor()` (`src/cli/serve/inference_backend.cpp:1127-1129`) задаёт `layered_prefill_pass_tokens = 4096` и `max_prefill_layer_groups = 16`, а `QwenPrefillYieldScope` (`inference_backend.cpp:1051-1066`) включает послойные уступки `QwenGpuExecutor::SetPrefillYield` (`src/models/qwen/hip/prefill.cpp:17-31`) между группами слоёв.
  - В `TextGenerationScheduler::Impl::RunYieldRound` (`src/cli/serve/text_generation_scheduler.cpp:1904-1963`) между группами слоёв уже допускаются новые запросы, обслуживается один принятый `short_peer` (`remaining <= decode_active_tokens`, 512) и шагают все активные декодеры.
- **Реальные пробелы в коде:**
  1. **Слепой FIFO в очереди до admission (`PopQueued`, `text_generation_scheduler.cpp:659-688`):** `queued_clients` выбирается строго по порядку клиентов (`queued_clients.front().requests.front()`). При занятых слотах (`C1` или `C4` под нагрузкой) короткий семейный/служебный запрос (например, 304 токена `aux_title_request` или вопрос на 500–1500 токенов) ждёт за длинными агентскими запросами (13K–100K токенов), которые забирают освободившийся слот первыми.
  2. **Фиксированный шаг уступок независимо от числа декодеров (`StepPrefill`, `text_generation_scheduler.cpp:1221-1258`):**
     - При `decoder_runnable == true` и `layered == true` (`Qwen`) `StepPrefill` всегда передаёт `yield_every_tokens = prefill_policy.decode_active_tokens` (512), что при 4096-токенном проходе даёт 8 групп слоёв (уступка каждые 5 слоёв из 40), хотя `max_prefill_layer_groups = 16` позволяет уступать вдвое чаще при `C4`.
     - Для не-слоистых путей (`layered == false`) и при нескольких активных декодерах (`others_decoding > 1`) чанк prefill также не уменьшается (всегда 512).

### 1.2. Доводка кэш-политики, удержание `System` при сжатии, `fe2694bc`/`8ab834a5` и `MemoryGovernor`

- **Что уже закрыто в `T2a-1` (`799cc613`, `664ed53e`):**
  - Изоляция `ram_only` (`IsRamOnlyChatRequest` до 1024 токенов финального промпта, защита обычных снимков и отказ от дискового кэша).
  - Диагностика `[cache] event=cache_prefix_loss` (`ContinuationPrefixMatch` и `LocatePromptPosition`).
- **Аудит апстрим-коммитов `fe2694bc` (#1104) и `8ab834a5` (#903):**
  1. **`fe2694bc` (`src/cache/retention.cpp:599-601`) — переносим точечно только сохранение выведенного branch point при `exact_replacement`:**
     - В `ContinuationCache::ReserveSnapshot` (`src/cli/serve/continuation_cache.cpp:1099-1100`) и `ContinuationCache::Commit` (`continuation_cache.cpp:1349-1351`) при точном переиздании записи (`exact_replacement`) сохраняется только явный `purpose == SnapshotPurpose::kBranchPoint`. Если же запись была `kContinuation`, от которой разошлись два потомка (`impl_->IsBranchPoint(target) == true`), то после удаления одного из потомков она теряет защиту точки ветвления. При `!ram_only` её нужно повышать до `SnapshotPurpose::kBranchPoint`.
     - **Не переносим** вторую часть `fe2694bc` (`stale` в `Victim` при `record_pressure`, `retention.cpp:471-483`): в `ContinuationCache` лимит записей `kMaxEntries = 128` (`src/cli/serve/text_model_runner.hpp:48`), снимки 35B-A3B весят сотни МиБ — единицы ГиБ, поэтому реальное ограничение — байтовый бюджет (`ram_only` уже уходит первым с rank `-1`), а допуск `kRetry`/`kHistory` к вытеснению `kContinuation` по счётчику оборотов для `ContinuationCache` не нужен.
  2. **`8ab834a5` (`src/core/hip/committed_backing.cpp`, `component_cache_resources.cpp`) — не переносим:** относится к `hipHostRegister` и сегментному пулу подсистемы `src/cache`. `ContinuationCache` и `QwenTextRunnerSnapshot` используют обычные host-буферы без `CommittedBackingPool`.
- **Реальный пробел при сжатии истории (compaction в Hermes Agent и `tests/functional/cache_compaction.py`):**
  - В `QwenChatTemplate::Render` (`src/models/qwen/chat_template.cpp:729-738`) рендерится единый системный блок `<|im_start|>system\n...<|im_end|>\n` (включая `reasoning`, `tools` и `system`/`developer`-сообщения), но его точная граница не экспортируется в `models::qwen::vision::Prompt` (`src/models/qwen/vision/prompt.cpp:324`) и `TextPreparedPrompt` (`text_model_runner.hpp:39-44`).
  - В результате, если в системе работает **один** активный диалог Hermes (без второго диалога-зонда `compaction_shared_probe`), то на границе `System+Tools` (~12K токенов, 55K символов) **вообще не создаётся чекпоинт**: первый чекпоинт ставится после первого `user`-сообщения, которое при сжатии истории удаляется!
  - Если экспортировать `system_prefix_tokens` и при холодном prefill (`system_prefix_tokens >= 512`, `!ram_only`) планировать RAM-чекпоинт в этой точке с `SnapshotPurpose::kBranchPoint`, то даже одиночный диалог Hermes удержит 12K-префикс в RAM (~640 МиБ при бюджете 16–32 ГиБ) и после compaction восстановится без перечитывания `System+Tools`.
- **Реальный пробел для `[идея strixite::MemoryGovernor]` (`src/cli/serve/text_model_runner.cpp:197-231, 586-601`, `continuation_cache.cpp:478-484, 955-1133`):**
  1. При старте в `QwenTextRunner::ResourceClaim()` (`inference_backend.cpp:1168-1169`) подряд вызываются `HostSnapshotBudgetBytes()` и `HostSnapshotCeilingBytes()`, что дважды открывает и парсит `/proc/meminfo` и `/proc/self/cgroup`.
  2. `ContinuationCache` читает `snapshot_support.capacity_bytes()` только один раз в конструкторе (`continuation_cache.cpp:480`). При внешнем росте потребления RAM на 128 ГБ UMA (`EXP-N`) предел RAM-кэша не снижается и новые снимки продолжают вызывать давление памяти вместо вытеснения старых снимков или пропуска захвата.

### 1.3. `GrammarCache` и оценка пер-запросного `capture`-лога

- **Факты по `GrammarCache`:**
  - В `src/core/json_constraint.cpp` кэши на уровне AST/автомата (`Compile` :1772, `ToolParameters` :1801, `WithReasoning` :1918, `WithTools` :1995) уже есть, но имеют два дефекта:
    1. Лимит `ToolParameters` равен **16**, тогда как `base_request.json` Hermes содержит **20** инструментов — на каждом запросе Hermes 20 инструментов выбивают друг друга из кэша и перекомпилируются заново.
    2. Все 4 кэша в `json_constraint.cpp` удаляют `cache.begin()` (лексикографически минимальный ключ в `std::map`), а не LRU.
  - В `TextModelRunner::BindConstraint` (`src/cli/serve/text_model_runner.cpp:764-782`) кэш `constraints_` (`TokenConstraint`, хранящий построенные 248K-маски словаря) вмещает всего **4** записи и удаляет FIFO (`erase(begin())`), поэтому одновременная работа Hermes + `aux_title_request` + OpenCode + MCP вызывает повторный обход 248K-trie.
- **Оценка пер-запросного `capture`-лога (`{prompt_n, tokens_n, tools, one_shot, drafted, accepted, emitted}`):**
  - В `T2a-1` уже развёрнуты гистограммы длин и попаданий кэша в `/metrics` (`gufo_request_prompt_tokens`, `gufo_request_cached_fraction`, `gufo_request_cache_total`, `gufo_active_long_requests`), а на каждый завершённый запрос `GenerationLogDetails` (`src/cli/serve/generation_metrics.hpp:271-319`) уже логирует `prompt_tokens`, `prefill_tokens`, `generated_tokens`, `draft_accepted`, `draft_proposed`, `draft_rounds`, `acceptance_pct`, `plan`, `batch_width` без текстов промптов.
  - Отдельный пошаговый файл/CLI-флаг `--capture-dir` нужен для калибровки `T7 cost gate` (Этап 2, **после** HIP-графов `T3–T5`, которые изменят стоимость шага `MTP1`). В `T2a-2` добавлять отдельный формат пошаговых файлов преждевременно — выносим в подготовку к `T7`.

### 1.4. Попутные мелкие закрытия

1. **Удаление мёртвой ветки из `767c6018` (`src/cli/serve/inference_backend.cpp:1621-1633`):**
   - В `QwenTextRunner::StreamPersistentSnapshot` первой строкой вызывается `PersistentSnapshotPayloadBytes(snapshot)` (`inference_backend.cpp:1542-1558`), которая уже делает `dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot)` и при `nullptr` (или несовпадении модели/полей) выбрасывает `std::invalid_argument("Qwen persistent snapshot does not belong to this runner")`.
   - Повторная проверка `if (qwen_snapshot == nullptr)` и комментарий из `004d8fa1` ниже недостижимы. Их удаление безопасно; отказ чужого снимка уже проверяется в `tests/models/qwen/persistent_snapshot_stream_test.cpp:290-307` (`ForeignSnapshot`).
2. **Актуализация `docs/faex2/ROADMAP_SPEC_01.md` (ревизия 4):**
   - Документ сейчас в ревизии 3 от `2026-11-17` с baseline `004d8fa1` и незавершёнными `T1`, `T2`, `T2a-1`, часовым прогоном памяти и 4-пользовательским профилем (реально закрыты на `be242d70` `2026-11-18`).

---

## 2. Предлагаемые изменения по точным файлам и классам

### Шаг А. Приоритет очереди и адаптивный prefill (`others_decoding`)

- **`src/cli/serve/text_generation_scheduler.hpp`:**
  - В `TextSchedulerPolicy`: добавить `std::size_t short_prompt_priority_tokens{2048};` (порог входных токенов для приоритета коротких запросов, `0` — отключено) и `std::size_t max_short_bypasses{4};` (максимум обгонов короткими запросами старейшего длинного запроса в очереди подряд — защита от голодания).
  - В `TextPrefillPolicy`: добавить `std::size_t min_decode_active_tokens{128};` (нижняя граница адаптивного чанка/шага уступок при нескольких активных декодерах).
- **`src/cli/serve/text_generation_scheduler.cpp`:**
  - В `TextGenerationScheduler::Impl`:
    - Валидировать `min_decode_active_tokens > 0 && min_decode_active_tokens <= decode_active_tokens` (если `min_decode_active_tokens == 0`, считать ошибкой; для дефолтных 512/128 корректно).
    - В `PopQueued(std::size_t free_slots)`:
      - Когда `queued_count <= free_slots` (все ожидающие запросы помещаются в свободные слоты сразу) или `short_prompt_priority_tokens == 0`, сохранять прежний честный round-robin по `queued_clients` (это сохраняет детерминизм одновременного допуска и совместимость с `TestShortArrivalBehindWaitingPrefill`).
      - Когда `queued_count > free_slots` (реальная конкуренция за дефицитный слот):
        - если `consecutive_short_bypasses >= scheduler_policy.max_short_bypasses`, принудительно взять старейший запрос из головы очереди по round-robin и сбросить `consecutive_short_bypasses = 0` (гарантия отсутствия голодания длинных запросов);
        - иначе искать первый короткий запрос (`request->prompt.size() <= scheduler_policy.short_prompt_priority_tokens`) в порядке round-robin от головы `queued_clients`; если голова очереди при этом была длинной и её обогнали — увеличить `consecutive_short_bypasses` (а если выбранный запрос из-за этого вынул не первый элемент `client.requests`, сохранить `client` на своём круговом месте либо сдвинуть в конец согласно round-robin); если коротких в очереди нет — взять голову очереди и сбросить `consecutive_short_bypasses = 0`.
    - В `StepPrefill(...)`: передавать число активных декодеров `others_decoding` (`decoding.size()`).
      - Вычислять `effective_active_tokens`: при `others_decoding <= 1` остаётся `prefill_policy.decode_active_tokens` (512); при `others_decoding >= 2` делить `decode_active_tokens` пропорционально (`std::max(prefill_policy.min_decode_active_tokens, prefill_policy.decode_active_tokens / others_decoding)` — для 512 и 2–4 декодеров это 256 / 170 / 128).
      - Для `Qwen` (`layered == true`) ширина прохода `layered_prefill_pass_tokens = 4096` **не уменьшается**, а `yield_every_tokens` становится `effective_active_tokens`, увеличивая число групп слоёв (например, с 8 до 16 при `max_prefill_layer_groups = 16`), т.е. уступки между слоями идут чаще без потери эффективности батча 4096. Для не-слоистого пути (`layered == false`) уменьшается сам `budget = effective_active_tokens` (идея `strixite::others_decoding`).

### Шаг Б. Удержание `System`-префикса, `fe2694bc` и `MemoryGovernor`

- **`src/models/qwen/chat_template.hpp` и `src/models/qwen/chat_template.cpp`:**
  - В `QwenChatTemplate::Render(...)` добавить необязательный выходной параметр `std::size_t* system_prefix_bytes = nullptr`, заполняемый позицией `output.size()` сразу после закрытия `<|im_start|>system\n...<|im_end|>\n` (строка 738), если системный блок непустой. Логика рендера и токенов не меняется.
- **`src/models/qwen/vision/prompt.hpp` и `src/models/qwen/vision/prompt.cpp`:**
  - В `models::qwen::vision::Prompt`: добавить `std::size_t system_prefix_tokens{0};`.
  - В `Prepare(...)`: по аналогии со `stable_prefix_tokens` (строки 318–324) вычислять `system_prefix_tokens` только при валидной токенной границе `system_prefix_bytes > 0` (когда суффикс после `system_prefix_bytes` совпадает с хвостом `prompt.tokens`, без дополнительной обработки изображений).
- **`src/cli/serve/text_model_runner.hpp`, `text_generation_scheduler.hpp`, `inference_backend.cpp`, `text_model_runner.cpp`:**
  - В `TextPreparedPrompt` и `TextGenerationScheduler::RequestMetadata`: добавить `std::size_t system_prefix_tokens{0};` (по умолчанию `0`, прямые вызовы не меняются). Пробросить из `PrepareQwenPrompt` -> `chat`/`start_chat` -> `Submit` -> `TextRunnerPool::Acquire(..., std::size_t system_prefix_tokens = 0)`.
  - В `TextRunnerPool::Request::Impl`: константа `kSystemCheckpointMinTokens = 512`. Если `!ram_only` и `system_prefix_tokens >= kSystemCheckpointMinTokens` и `system_prefix_tokens < prompt.size()` и `system_prefix_tokens > prefill_offset` и `!lease.HasSnapshotFor(prompt.first(system_prefix_tokens))`, добавлять `system_prefix_tokens` в `checkpoints` и помечать его как системную точку ветвления (`branch_checkpoints`), чтобы при достижении границы `CaptureBoundarySnapshot` сохранил его в RAM с `SnapshotPurpose::kBranchPoint` (без принудительной записи на диск на одиночном холодном старте; дисковый `shared_prefix` сохраняет прежнюю политику при появлении второго диалога).
- **`src/cli/serve/continuation_cache.hpp` и `src/cli/serve/continuation_cache.cpp`:**
  - **Часть `fe2694bc`:** в `ReserveSnapshot` (при удалении `exact`, строки 1098–1102) и в `Commit` (`exact_replacement`, строки 1349–1351) для `!ram_only` сохранять статус `SnapshotPurpose::kBranchPoint`, если `entry.purpose == SnapshotPurpose::kBranchPoint || impl_->IsBranchPoint(target)`.
  - **Идея `MemoryGovernor` (`continuation_cache.hpp/.cpp` + `text_model_runner.hpp/.cpp` + `inference_backend.cpp`):**
    - В `text_model_runner.hpp/.cpp`: объединить расчёт доступной хост-памяти в `HostSnapshotLimits()` (одно чтение `/proc/meminfo` и `/proc/self/cgroup`), через неё реализовать `HostSnapshotBudgetBytes()`, `HostSnapshotCeilingBytes()` и вызов в `QwenTextRunner::ResourceClaim()`.
    - В `ContinuationCache::SnapshotSupport`: добавить опциональный callback `SnapshotCapacity ceiling_bytes;` (по умолчанию пустой, старые тесты не затрагиваются). В `MakeSnapshotSupport` (`text_model_runner.cpp`) при `options.capacity_bytes == 0` или с учётом потолка он вызывает `HostSnapshotCeilingBytes()` (единое чтение `/proc/meminfo` за один вызов `ReserveSnapshot`).
    - В `ContinuationCache::ReserveSnapshot` (`continuation_cache.cpp`): один раз до цикла `while` вычислять `effective_capacity = snapshot_support.ceiling_bytes ? std::min(snapshot_capacity_bytes, snapshot_support.ceiling_bytes()) : snapshot_capacity_bytes;` (без повторного чтения `/proc/meminfo` на итерациях вытеснения), и проверять `fits` / `can_fit_after_eviction` относительно `effective_capacity` (вытесняя лишние старые снимки в рамках того же цикла при снижении свободной RAM).

### Шаг В. `GrammarCache` (LRU + размер под 20 инструментов Hermes)

- **`src/core/json_constraint.cpp`:**
  - Вынести общий компактный шаблон `BoundedLruCache<Key, Value>` в анонимном пространстве файла (`Get` поднимает ключ на вершину LRU, `Put` вытесняет давно не использованную запись).
  - Перевести кэши `JsonConstraint::Compile` (16 записей), `JsonConstraint::ToolParameters` (увеличить лимит с `16` до **64**, чтобы 20 инструментов Hermes и соседние схемы OpenCode не вытесняли друг друга), `JsonConstraint::WithReasoning` (16), `JsonConstraint::WithTools` (16) и `TokenConstraint::masks_` (16, сохраняя защиту `initial_state_`) на `BoundedLruCache`.
- **`src/cli/serve/text_model_runner.hpp` и `src/cli/serve/text_model_runner.cpp`:**
  - В `TextModelRunner::BindConstraint` (`text_model_runner.cpp:764-782`): заменить вектор из 4 записей с FIFO-удалением на LRU-список/вектор до **16** записей (`kMaxBoundConstraints = 16`), поднимая найденный `TokenConstraint` в хвост (самый свежий) и удаляя наименее недавно использованный.

### Шаг Г. Попутные мелкие закрытия

- **`src/cli/serve/inference_backend.cpp:1621-1633`:**
  - В `QwenTextRunner::StreamPersistentSnapshot` убрать недостижимую ветку `dynamic_cast` + `if (qwen_snapshot == nullptr)` и временный комментарий из `004d8fa1`, заменив на `const auto& qwen = static_cast<const QwenTextRunnerSnapshot&>(snapshot);` после `const auto expected_bytes = PersistentSnapshotPayloadBytes(snapshot);`.
- **`docs/faex2/ROADMAP_SPEC_01.md` (ревизия 4):**
  - Обновить заголовок и статус на **ревизию 4** (прод `be242d70` / `964e84e5…`, резерв `004d8fa1` / `94a09930…`, ядро `7.3.4-1-cachyos`).
  - В §1.1, §1.2 п.6, §2.В п.8 и 11, §4 (Этап 0, Этап 1):
    - отметить как **завершённые** (2026-11-17..18): `T1` (`POST /tokenize`), `T2` (`input_tokens`), `T2a-1` (`ram_only`, `/metrics` распределения длин/кэша, `cache_prefix_loss`, live smoke 20/20), часовой прогон памяти (`be242d70`: RSS +0,11 ГиБ, GTT +0,07 ГиБ, 0 полных перечитываний; долг `004d8fa1` закрыт по содержанию при сохранении оговорки «час ≠ 24/7`), 4-пользовательский профиль (`187,2 ток/с` суммарно, `MTP 83–93 %`) и независимое ревью `T2a-1` (`APPROVE`);
    - в Этапе 0 п.2 оставить реальные открытые пункты: историческое ревью коммитов варианта Б (`158711ad`, `004d8fa1`), статус `r3-speed` (`UNMATCHED_WORK`), проверку реального автозапуска `gufo-faex2.service` при плановой перезагрузке (а удаление ветки `767c6018` закрыть в рамках `T2a-2`);
    - в Этапе 1 оставить реальные открытые долги: `T2a-2` (приоритеты/адаптивный prefill, `System`-чекпоинт при compaction, `MemoryGovernor`, `GrammarCache`), прикладной набор 50–100 задач семейного чата, сбор статистики `/metrics` в эксплуатации + решение по `capture` к `T7`, контракт точности `Q1` и тест зрения `model-vision-serving-mtp` (35/36 → 36/36), опциональный даунскейл крупного текста 48–96 pt.

---

## 3. Инварианты и границы безопасности

1. **Изоляция `ram_only` не ослабляется:** `ram_only` запросы (<= 1024 токенов) не планируют `system_prefix_tokens` чекпоинтов, не повышают `kBranchPoint`, не обращаются к `disk_store` и вытесняются раньше любых обычных снимков.
2. **Ограничение GPU-слотов `C4` сохраняется:** новые слоты/арены GPU не создаются; в тестах `SessionStates` остаётся согласованным.
3. **Приватность логов и метрик:** ни в планировщике, ни в кэше, ни в `GrammarCache` не логируются тексты сообщений, JSON-схемы или ID токенов.
4. **Честное планирование без голодания:** короткие запросы могут обгонять длинный запрос в очереди не более `max_short_bypasses` раз подряд; при одновременном наличии достаточных слотов (`queued_count <= free_slots`) порядок допуска не нарушается; гарантии `SetPrefillYield` и `RunYieldRound` сохраняются при `others_decoding <= 1`.
5. **Правовая чистота и сохранность `src/cache`:** 0 строк кода из `strixite` (GNU AGPL-3.0); вся реализация пишется с нуля под архитектуру `gufo-faex2`. Подсистема `src/cache` и бинарный формат дисковых снимков не изменяются.

---

## 4. План проверок на Фазе 2 (до любого живого окна)

1. **CPU / unit-тесты (`cmake --preset cpu-test` и `cmake --preset cpu-sanitizer` ASan/UBSan):**
   - `text_generation_scheduler_test`:
     - приоритет коротких запросов при конкуренции за слоты (`queued_count > free_slots`), срабатывание защиты от голодания длинного запроса (`max_short_bypasses`), отключение флагом `0`, сохранение всех существующих тестов включая `TestShortArrivalBehindWaitingPrefill`;
     - адаптивное уменьшение эффективного чанка / рост `layer_groups` при `others_decoding >= 2` как для `layered`, так и для не-слоистого runner, и точное сохранение прежнего поведения при `others_decoding == 1`.
   - `continuation_cache_test` + `text_model_runner_test`:
     - сохранение/повышение выведенного `IsBranchPoint` при `exact_replacement` (`!ram_only`) и защита от `ram_only`;
     - удержание `system_prefix_tokens` в одиночном диалоге при имитации сжатия истории (холодный prefill `[system(600), user_1]` -> рост до 10K -> compaction `[system(600), summary]` -> точный hit `600` без второго диалога-зонда);
     - `MemoryGovernor`: однократный вызов `ceiling_bytes` за `ReserveSnapshot`, принудительное вытеснение старых снимков при падении потолка памяти хоста и отказ нового резерва без превышения потолка, а также unit-тест `HostSnapshotLimits()` в `text_model_runner_test`;
     - `BindConstraint` LRU до 16 записей в `text_model_runner_test` (вытеснение неиспользуемого, а не горячего).
   - `json_constraint_test`:
     - кэширование >= 20 различных `ToolParameters` схем без взаимного вытеснения; LRU-порядок вытеснения в `Compile` / `WithTools` / `TokenConstraint::Allowed` (горячая схема переживает серию редких).
   - `qwen_chat_template` / `prompt_test`:
     - точная граница `system_prefix_bytes` (и `system_prefix_tokens`) при наличии `system` + `tools`, отсутствии `system`, изменении порядка сообщений; побайтное равенство отрендеренного текста и токенов.
   - Полный прогон `cmake --build --preset pr` (все hosted CPU-контракты) + `http_server_test`, `openai_chat_test`, `serve_cli_test`.
2. **Форматирование, документация и Python-регрессии:**
   - `python3 tools/ci/check-format.py`;
   - `python3 tests/tools/check_docs_test.py` и проверка всех markdown-ссылок;
   - `PYTHONPATH=tests/functional python3 -m unittest tests/functional/test_runner.py tests/functional/test_hermes_synth.py tests/functional/test_dflash2_eval.py`.
3. **HIP-компиляция (без запуска моделей и остановки продакшена `be242d70`):**
   - сборка `gufo` (`ENGINE_ENABLE_HIP=ON`, `GUFO_ENABLE_NPU=OFF`, `gfx1151`) + компиляция `qwen_persistent_snapshot_stream_test` и `qwen_persistent_snapshot_mtp_stream_test` (проверка удаления мёртвой ветки `767c6018` и интеграции `PrepareQwenPrompt`).

---

## 5. Что сознательно НЕ делаем в `T2a-2` и почему

1. **Отдельный пошаговый `--capture-dir` лог:** перенесён в подготовку `T7 cost gate` (Этап 2 после HIP-графов `T3–T5`), чтобы не фиксировать формат до определения точных признаков модели стоимости и не создавать новый файловый вывод на диске.
2. **Правило `stale` при `record_pressure` из `fe2694bc` и сегментный `hipHostRegister` из `8ab834a5`:** `8ab834a5` относится к другой подсистеме (`src/cache`), а `stale` обходит защиту долгоживущих префиксов ради проблемы `kMaxEntries=128`, неактуальной при размерах снимков Qwen3.6 и изоляции `ram_only`.
3. **Квота записи на диск `write_gib_per_hour` для `ContinuationDiskStore`:** после `ram_only` (`T2a-1`) и существующего `min_checkpoint_step_tokens = 2048` очередь записи на штатном 64 ГиБ дисковом кэше составляет 0 ГиБ.
4. **Новые CLI-флаги для `T2a-2`:** все пороги задаются в `TextSchedulerPolicy` / `TextPrefillPolicy` с безопасными значениями по умолчанию без раздувания CLI.
