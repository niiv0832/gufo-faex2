# T2a-1 — план реализации защиты кэша Hermes и телеметрии

> **Для исполнителя:** выполнять задачи последовательно с
> `executing-plans` (самостоятельно) или `subagent-driven-development`
> (если владелец выберет делегирование). Состояние отмечать флажками.

**Цель:** служебный structured one-shot не вытесняет длинные RAM-снимки
и не использует disk store; `/metrics` и журнал показывают длины/reuse/потери.

**Архитектура:** внутренняя политика проходит от окончательной подготовки
chat через scheduler/runner до `ContinuationCache`. Политика хранения
не меняет identity и формат снимков. Телеметрия использует числовые агрегаты
и единичную terminal-запись, без сохранения текстов/ID токенов.

**Стек:** C++20, существующие fake runner/CTest, Python unittest,
CMake/Ninja; Linux x86-64, TheRock HIP `gfx1151` только для compile/link.

**Дизайн:** [согласованная спецификация](../specs/2026-11-17-hermes-cache-telemetry-design.md).
Исходная документационная контрольная точка — `56fbd126` ветки `faex2`.

## Общие ограничения

- `strixite` — только архитектурные идеи, 0 строк копирования/перевода кода.
- Не менять `src/cache`, `src/core/hip/committed_backing.*`, GPU-ядра,
  арифметику и `continuation_disk_store.cpp`/дисковый формат.
- `ram_only`: `tools.empty()`, одно последнее user, перед ним только
  system/developer, нет tool history/images, ненулевой response_format,
  окончательный промпт ≤1024 токенов.
- Длина — `std::size_t` **после** `ConstrainChatRequest` → `PreparePrompt`.
- RAM-only не использует disk store вообще, включая Restore/Touch.
- Обычные снимки защищены и при reserve, и при commit, без downgrade.
- Занятый GPU lease не отбирается; C1/C4 ограничены `--sessions`.
- Новые histogram/counters — только после успешного Acquire; gauge —
  существующие processing-слоты, включая cache preparation/parked followers.
- Пороги диагностики: `>512` для lost/unreused, `>=8192` для prefill WARN.
- Старые метрики, legacy Python `COUNTERS`/`TYPES` и точный JSON
  `input_tokens` не менять/не ослаблять ради новых series.
- Стенд — только собственный временный каталог, без остановки/настройки
  сервиса, без моделей, GPU-тестов и HTTP smoke.
- Коммиты на русском, Conventional Commit, существующий noreply,
  только собственные пути; CONTEXT/CHANGELOG обновляются с каждым шагом.
- Push/PR, sudo, установка пакетов, rebase/amend/reset/clean запрещены.

## Особое внимание при проверке

1. Финальная системная схема пересекает порог 1024, хотя сообщения короткие:
   классификатор должен оставить обычную политику — задача 1.
2. Два lease одновременно резервируют/публикуют одну запись: aux не должен
   вытеснить обычную запись через exact replacement — задача 1.
3. Ни один вызов disk store от aux, включая live-checkpoint до Request::Impl;
   старый повреждённый файл и его timestamps не меняются — задача 1.
4. Parked follower отменён без lookup: gauge был ненулевым, но нового
   histogram/miss observation нет — задача 2.
5. Неподдержанный шаблон или image identity mismatch: сообщение/причина
   не выдумываются, токены/тексты не попадают в лог — задача 3.

## Карта файлов

| Файлы | Ответственность |
|---|---|
| `src/cli/serve/continuation_cache.{hpp,cpp}` | policy lease/records, RAM admission/eviction, ближайший prefix-match |
| `src/cli/serve/text_model_runner.{hpp,cpp}` | CPU-классификатор, передача policy, отсутствие disk I/O, optional позиция сообщения |
| `src/cli/serve/text_generation_scheduler.{hpp,cpp}` | metadata policy, однократный terminal observation/log |
| `src/cli/serve/inference_backend.cpp` | оба chat-пути: final length/policy; Qwen hook управляющего токена |
| `src/cli/serve/text_generation_backend.hpp` | числовые diagnostic result-поля |
| `src/cli/serve/generation_metrics.hpp`, `http_server.cpp` | bounded aggregates и Prometheus exposition |
| `tests/cli/{continuation_cache,text_model_runner,text_generation_scheduler,http_server}_test.cpp` | fake-based RED/GREEN, без модели |
| `tests/functional/test_runner.py` | совместимость legacy parser/строгого count JSON |
| `docs/SERVER.md` | добавочный контракт метрик, policy и privacy |

Новых translation units и новых dependencies не нужно. Targets уже
зарегистрированы в корневом `CMakeLists.txt`; его не менять без необходимости.
Данные снимков и сериализация остаются прежними.

## Подготовка исполнения и доказательств

- [ ] Загрузить выбранный workflow и `using-git-worktrees`; создать **свой**
  изолированный worktree от текущей `faex2`, не менять чужие worktree.
  Зафиксировать base SHA и исходный чистый status.
- [ ] Перед первым SSH перечитать текущий эксплуатационный контекст.
  Read-only preflight: service active, PID/identity рабочего бинарника,
  disk/RAM, compiler/CMake/Ninja/clang-format, отсутствие конфликтующей сборки.
  Никаких generation-запросов к рабочему серверу.
- [ ] После короткого уведомления создать на стенде только
  `$HOME/gufo-t2a1-verify-20261117-r1/{src,tmp,out}`; при существующем корне
  остановиться и выбрать новое имя, не удалять/перезаписывать старый прогон.
  Сохранить SHA `/proc/<child>/exe` и status обоих штатных исходных клонов.
- [ ] Передавать только tracked source + собственные текущие diff, без `.local`,
  секретов, AGPL-исходников и артефактов. RED/GREEN версии фиксировать
  отдельными patch/hash manifests; не перезаписывать логи неуспешных прогонов.
- [ ] На стенде `TMPDIR=$RUN/tmp`, `CMAKE_BUILD_PARALLEL_LEVEL=4`,
  `nice -n 19`, устойчивый запуск в собственной `tmux`-сессии.
  Вывод каждого шага — отдельный файл в `$RUN/out/`.
  Дождаться автоматического уведомления фоновой работы, не запускать
  дубликаты suite, не принимать отсутствие FAILED-строк за успех.

## Задача 1. `ram_only`: классификатор, RAM-политика, изоляция от диска

**Файлы:** cache/runner/scheduler/backend из карты; тесты cache/runner/scheduler;
`docs/SERVER.md`, CONTEXT/журнал форка.

**Интерфейсы:**
- Добавить `bool IsRamOnlyChatRequest(const ChatRequest& original,
  std::size_t prepared_prompt_tokens) noexcept` в `text_model_runner.hpp/.cpp`;
  `inline constexpr std::size_t kRamOnlyMaxPromptTokens = 1024`.
- Добавить `bool ram_only{false}` в scheduler RequestMetadata/ScheduledRequest.
- Последний аргумент основных `TextRunnerPool::Acquire` и
  `ContinuationCache::Acquire` — `bool ram_only = false`.
- Lease и snapshot/live records хранят признак раздельно от `SnapshotPurpose`.
  Передать его через private ReserveSnapshot/Commit и move-операции lease.
- В `GenerateScheduled` последний аргумент `bool ram_only = false`;
  `chat`/`start_chat` вычисляют значение по original request и
  `prompt->tokens.size()` **до** `std::move`.

- [ ] Написать `TestRamOnlyChatClassification`: system+user/schema/304 → true,
  json_object → true; no format/tools/assistant/tool history/images/
  tool_calls/неверный enum role/два user → false; 1024 → true, 1025 → false.
  Для 64-bit `size_t` проверить значение выше UINT32_MAX без выделения вектора:
  классификация false, нет усечения до короткой длины.
- [ ] Написать `TestRamOnlyUsesFinalPreparedLength`: fake renderer учитывает
  framing_suffix; после реального ConstrainChatRequest и PreparePrompt
  конечные 1024/1025 дают разные policy при одинаково коротком user content.
  Не вычислять размер отдельно по исходным сообщениям.
- [ ] Написать cache-регрессии `TestRamOnlyKeepsLongConversation` и
  `TestRamOnlyAdmissionDoesNotDowngradeOrdinary`: synthetic 55K tokens,
  малые fake payload, ограничение records и bytes отдельно, серия aux,
  ordinary exact match, retry/history/branch-point, promotion, два lease
  с reserve→публикацией обычной записи→commit aux.
- [ ] Написать runner-регрессии `TestRamOnlyNeverUsesDiskStore`:
  первая/повторная генерация, отказ RAM, boundary/shared/live captures,
  Cancel/throw; queued bytes=0, persistence capture/restore count=0,
  старые bytes/mtime сохранены, aux повторяется из RAM при наличии места.
  Обычный запрос в том же pool по-прежнему persists/restores через новый pool.
- [ ] RED: собрать и выполнить `continuation_cache_test`,
  `text_model_runner_test`, `text_generation_scheduler_test` по командам ниже.
  Сначала отсутствие новых интерфейсов допустимо как compile-RED,
  но eviction/disk регрессии дополнительно должны падать на наблюдаемом
  старом поведении с временно отключённой policy, не только на missing symbol.
- [ ] Реализовать classifier и передачу policy. В `TextRunnerPool::Acquire`
  вычислить request-local disk store (`nullptr` для aux) **до** всех disk
  путей, включая live capture, restore и shared-boundaries.
- [ ] Реализовать общий eligibility guard жертвы для Reserve/Commit:
  aux конкурирует только с aux; ordinary не downgrade; aux не делает ordinary
  branch избыточным; exact/edited-tail/source обходы соблюдают guard.
  Обычные запросы сначала вытесняют aux. Не менять общий бюджет памяти.
- [ ] GREEN: повторить те же tests, затем format-check до C++ коммита.
  Проверить возврат всех reservations при cancel/throw и C1/C4 busy leases.
- [ ] Документировать policy/ограничения; обновить CONTEXT/журнал.
  Коммит: `feat(cache): защищены длинные снимки от служебных запросов Hermes`.

## Задача 2. Гистограммы `/metrics` и совместимость парсеров

**Файлы:** generation_metrics/http_server/scheduler и их тесты,
`tests/functional/test_runner.py`, `docs/SERVER.md`, CONTEXT/журнал.

**Интерфейсы:**
- `void RecordCacheRequestMetrics(const TextGenerationBackend::Result&)`
  в `generation_metrics.hpp`; отдельный от прежних token/time counters.
- `void WriteCacheRequestMetrics(std::ostream&)` там же: process aggregates.
- `void WriteActiveLongRequestMetrics(std::ostream&,
  std::span<const TextGenerationBackend::SessionState>)` там же.
- ScheduledRequest: `bool cache_lookup_complete{false}` и
  `bool cache_metrics_recorded{false}`; отмечать lookup только после
  успешного Acquire, наблюдать один terminal result, ошибки исключить.

- [ ] Написать scheduler-тест `TestCacheRequestDistributions`: длины
  `512/513/2048/8192/16384/32768/65536/131072`, доли
  `0/.25/.5/.75/.9/.99/1`, cached>prompt; отсутствие деления на ноль.
  Сравнивать process-wide **дельты**, не абсолютные totals между тестами.
- [ ] Добавить cancel/ошибка Acquire/parked follower/no Wait/repeated Wait
  регрессии: cache observation отсутствует до lookup, однократен после lookup,
  failure не учитывается; прежний token/time accounting не меняется.
- [ ] Написать HTTP-тест `TestCacheDistributionMetrics`: HELP/TYPE,
  cumulative `_bucket`/`_sum`/`_count`, все фиксированные cache labels
  (`miss`, `memory`, `disk`), нули без модели. Gauge строго `>` на
  `16384/32768/65536`, только processing; generated tokens не влияют.
  Один session snapshot для новых gauges и существующего KV-ratio.
- [ ] В `TestOpenAiInputTokens`/`TestLlamaTokenize` проверить нулевые дельты
  **новых** агрегатов и прежних counters после позитивных/негативных count
  запросов; точный JSON count не расширяется.
- [ ] Добавить Python-тест
  `test_prometheus_additional_cache_series_preserve_legacy_accounting`:
  unknown HELP/TYPE + labelled histogram/cache/gauge series игнорируются,
  legacy parse_metrics результат и нулевые COUNTERS-дельты сохраняются.
  Сохранить негативные проверки missing/duplicate/incorrect TYPE и строгий
  `test_openai_count_tokens_contract` из `metrics.summarize`.
- [ ] RED: выбранные C++ тесты; Python compatibility-тест может сразу быть
  GREEN, так как parser уже допускает новые метрики — это фиксация контракта,
  а не повод искусственно портить parser. RED нужен у новой instrumentation.
- [ ] Реализовать bounded atomics: length buckets
  `512,2048,8192,16384,32768,65536,131072,+Inf`, fraction buckets
  `0,.25,.5,.75,.9,.99,1,+Inf`. Нет labels с пользовательскими данными.
  Экспортировать `gufo_request_prompt_tokens`, `gufo_request_cached_fraction`,
  `gufo_request_cache_total`, `gufo_active_long_requests`.
- [ ] GREEN: C++ и Python команды ниже, format-check; legacy `TYPES`/`COUNTERS`
  и production counting handlers менять только при доказанном дефекте,
  не ради принятия новых полей. Метаданные не попадают в count JSON.
- [ ] Обновить SERVER/CONTEXT/журнал; коммит:
  `feat(metrics): добавлены распределения длин запросов и попаданий кэша`.

## Задача 3. Диагностика крупных потерь префикса

**Файлы:** cache/runner/backend result/scheduler/generation_metrics и тесты
из карты, Qwen serve-адаптер в `inference_backend.cpp`, docs.

**Интерфейсы:**
- `ContinuationPrefixMatch { std::size_t common_prefix_tokens{0};
  std::size_t nearest_tokens{0}; bool input_matches{false}; }` в cache.hpp.
- `Lease::prefix_match() const noexcept` и
  `TextRunnerPool::Request::cache_prefix_match() const noexcept`;
  Result получает `ContinuationPrefixMatch cache_prefix_match`.
  Старые `ContinuationLookup` и miss-поля сохраняют прежнюю семантику.
- `TextModelRunner::MessageStartToken() const -> std::optional<TextRunnerToken>`
  с default nullopt; Qwen override в serve-адаптере получает один ID
  через `TokenizeControl(tokenization::kImStart)`, иначе nullopt.
- `TextPromptPosition`: `Kind {kUnknown,kBeforeMessages,kMessage,kEnd}`,
  `kind`, `std::size_t message_index{0}`, `std::size_t token_offset{0}`.
  Определение размещается в `text_generation_backend.hpp` перед Result,
  чтобы не создать цикл backend.hpp → runner.hpp → backend.hpp.
  `LocatePromptPosition(std::span<const TextRunnerToken>, std::size_t at,
  std::optional<TextRunnerToken> message_start) -> TextPromptPosition`
  в runner.hpp/.cpp. Result: `TextPromptPosition cache_divergence`.
  Для `ContinuationPrefixMatch` backend.hpp подключает cache.hpp,
  у которого нет зависимости от backend/runner.
- `CachePrefixLossLogDetails(const Result&) -> std::string` в metrics.hpp:
  пустая строка при отсутствии события; числовые/fixed-name поля иначе.

- [ ] Написать `TestPrefixMatchSurvivesPartialHitAndDiskRestore`:
  cold/no record, full hit, short hit + nearest deep record, truncate,
  identity mismatch и disabled reuse. Ближайшая запись фиксируется до
  очистки live_tokens; disk restore не очищает новый prefix_match.
  LCP tie: совместимая identity, затем более длинная запись.
- [ ] Написать `TestPrefixLossThresholdsAndPrivacy`: `lost=512/513`,
  `unreused=512/513`, `prefill=8191/8192`, common=0 с changed input,
  full exact hit без события; once terminal/cancel, cache disabled молчит.
  `lost=nearest-common`, `unreused=max(common-cached,0)`.
  Запрет выводить secret marker, prompt/array/token ID или содержимое tools.
- [ ] Написать `TestPromptDivergencePosition`: начало/середина сообщения,
  до первого, конец, nullopt hook; независимые synthetic token fixtures.
  Индекс с нуля по rendered сообщениям, не по исходному JSON.
- [ ] RED на выбранных cache/runner/scheduler tests; затем минимально
  реализовать новый диагностический scan без копии токенов и без Decode.
  Не выводить guessed cause tools/system/compaction.
- [ ] После Acquire сохранить prefix_match и позицию до перемещения/потери
  prompt; terminal-лог один раз на result, включая частичный hit.
  INFO при lost/unreused>512 и common>0, WARN при changed prefix/input +
  фактическом prefill>=8192. Не использовать отмену до lookup как cold miss.
- [ ] GREEN + format, compile HIP-path; обновить docs/CONTEXT/журнал.
  Коммит: `feat(cache): добавлена диагностика крупных потерь префикса`.

## Команды RED/GREEN и итоговой проверки

Все команды на стенде выполняются из **собственного** `$RUN/src`,
с отдельными stdout/stderr/exit status в `$RUN/out/<phase>/`.
`set -euo pipefail` обязателен в worker, чтобы не повторить ложный BUILD_OK
старого wrapper. Никакие старые build/restart wrappers не запускать.

Итерация только затронутых целей после первого configure:

```bash
cmake --preset cpu-test
cmake --build --preset cpu-test --parallel 4 --target continuation_cache_test text_model_runner_test text_generation_scheduler_test http_server_test
ctest --preset cpu-test -R '^(continuation_cache_test|text_model_runner_test|text_generation_scheduler_test|http_server_test)$' --output-on-failure
python3 tools/ci/check-format.py
PYTHONPATH=tests/functional python3 -m unittest tests/functional/test_runner.py
```

Ожидается: каждый selected test passed, отсутствие format diff, Python OK.
В RED зафиксировать конкретную ожидаемую assertion/ошибку, не принимать
проблему toolchain/link/dependency за воспроизведение cache-дефекта.

- [ ] Итоговая CPU-проверка: `cmake --preset cpu-test`, затем
  `cmake --build --preset pr --parallel 4`; прочитать полный итог CTest,
  фактическое число тестов и exit status, не переносить прежние «56/56».
- [ ] `cmake --preset cpu-sanitizer`; сборка:
  `cmake --build --preset cpu-sanitizer --parallel 4 --target continuation_cache_test continuation_disk_store_test text_model_runner_test text_generation_scheduler_test http_server_test`.
  Запуск:
  `ctest --preset cpu-sanitizer -R '^(continuation_cache_test|continuation_disk_store_test|text_model_runner_test|text_generation_scheduler_test|http_server_test)$' --output-on-failure`.
  Ожидается 5 passed без ASan/UBSan reports.
- [ ] HIP compile/link без исполнения:

```bash
R="$HOME/tools/therock-10.1.0"
export ROCM_PATH="$R" HIP_PATH="$R"
export PATH="$R/bin:$R/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$R/lib:$R/lib/llvm/lib"
cmake --preset release -B build/t2a1-hip -DCMAKE_PREFIX_PATH="$R" -DCMAKE_HIP_COMPILER="$R/lib/llvm/bin/clang++" -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 -DENGINE_ENABLE_HIP=ON -DGUFO_ENABLE_NPU=OFF
cmake --build build/t2a1-hip --parallel 4 --target gufo
```

  Ожидается configure/build exit 0, HIP=ON/NPU=OFF/gfx1151 в CMakeCache и
  compile_commands. TheRock path сверить в preflight; при несовпадении
  остановиться, не устанавливать toolchain и не менять систему.
- [ ] Python регрессии:
  `PYTHONPATH=tests/functional python3 -m unittest tests/functional/test_runner.py tests/functional/test_hermes_synth.py tests/functional/test_dflash2_eval.py`.
  Модельный `openai_sdk --suite openai-count-tokens` **не запускается**;
  CPU проверяет его parser/contracts, не объявлять это живой HTTP-приёмкой.
- [ ] Повторить format после всех C++ правок; `git diff --check`,
  проверка markdown ссылок изменённых docs через `check-docs` API.

## Завершение, перенос и очистка

- [ ] Проверить требования дизайна/обоих уточнений владельца по diff и
  сохранённым RED/GREEN логам; независимая whole-branch проверка после
  выбранного способа исполнения. Не удалять временные сборки до её окончания.
- [ ] После устранения замечаний без скрытого ослабления тестов —
  повторить только затронутые проверки; сохранить также исходные FAIL.
- [ ] Перенести свои логические коммиты в `faex2` штатным fast-forward/
  cherry-pick по выбранному worktree workflow, без rebase/amend и без
  переноса чужих изменений. Проверить author/committer noreply.
- [ ] Эксплуатационные документы обновить **после фактических** действий:
  PROJECT_CONTEXT §0.1 и CHANGELOG; логи/manifest — в отдельном results
  эксплуатационного проекта. Его Git-идентичность не менять.
- [ ] Удалить только свои `$RUN/src` (он включает build) и `$RUN/tmp`,
  при наличии отдельного `$RUN/build` — его тоже. Проверить обычные пути,
  отсутствие symlinks/живых workers; сохранить `$RUN/out`/manifest/logs.
  Не использовать `git clean` или удаление широких родительских каталогов.
- [ ] Сверить service/PID/identity и SHA рабочего процесса, status двух
  штатных клонов до/после; при незапланированной перемене остановиться
  и сообщить, не перезапускать сервер «для проверки».
- [ ] Удалить только свой временный worktree через `git worktree remove`;
  итоговый status `faex2` чистый, коммиты локальные, push не выполнен.
- [ ] Итог: точные SHA, команды/counts/результаты, пути raw,
  подтверждение cleanup и пределы: нет GPU/model/HTTP/performance приёмки,
  C4 slot-limit остаётся, deploy не выполнен.

## Самопроверка плана

- [x] Требования дизайна покрыты задачами 1–3 и итоговыми проверками.
- [x] Сигнатуры связанных задач согласованы; defaults сохраняют старые callers.
- [x] Пять рисков вынесены в конкретные тесты.
- [x] Нет запланированных изменений формата/ядер/сервиса или скрытого smoke.
- [x] Уточнения владельца о final length, Python parser и C4 учтены.
- [ ] Владелец подтвердил план и выбрал способ исполнения.
