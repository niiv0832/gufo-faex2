# План потоковой сериализации снимков Qwen

> Исполнение: `superpowers:executing-plans` или `superpowers:subagent-driven-development`, с проверкой каждого шага. Это продолжение порученных доработок аудита, не разрешение deployment.

**Цель:** убрать полный временный host-буфер записи Qwen snapshot, сохранив побайтный persistent format и параллельное восстановление.

**Архитектура:** Qwen выдаёт неизменяемый snapshot через существующий sink. GPU-области копируются последовательно через один scratch не больше 16 MiB; уже принадлежащие снимку host-данные MTP/logits выдаются без полной копии. Существующий disk store сохраняет checksum, точный byte count, fsync и atomic rename.

**Стек:** C++20, HIP gfx1151, существующие CMake targets и Linux toolchain. Системные пакеты не устанавливаются и не обновляются.

**Основания:** [SPEC](SPEC.md), [DESIGN](DESIGN.md), [итог final3](CLOSEOUT_2026-11-13.md#ordinary-final3-2026-11-14-0408-msk), [source review](../../.local/remed-20261113/codex-closeout/final3-rss-host-buffer-source-review.md), [существующая trace](../../.local/remed-20261113/codex-closeout/FINAL3_EXISTING_HOST_BUFFER_TRACE_ADDENDUM.md).

## Доказанный механизм и границы

До исправления `TextModelRunner::StreamPersistentSnapshot` выделял vector на весь payload; Qwen не переопределял метод. `ReadImage` отдельно держит полный файл. В diagnostic cache3 эти буферы размером 2865305893 и 2859404591 байт перекрывались 1,480035 с; RSS вырос и затем снизился. В cache4/cache5 такого перекрытия нет. Ordinary final3 имеет согласующийся по масштабу пик, но его точная allocation-атрибуция по INFO не доказана. Его **FAIL_MEMORY** остаётся: RSS max +2508189696 байт при пороге 1073741824.

Выбор исполнителя в рамках порученной доработки — настоящая потоковая запись. Общая блокировка Save/Restore добавляла бы ожидание и меняла поведение кэша. Уменьшение budgets не устраняет лишнюю копию. Потоковое чтение потребовало бы более широкого изменения checksum-before-mutation и не входит в этот шаг.

## Ограничения

- Не менять file/payload version, compatibility identity, арифметику, формат KV/SSM, порядок байтов, snapshots/rollback/pending tokens.
- Не менять RAM freeze/publication, queue ownership, min-step, LRU, бюджеты, критерии часа или пороги времени 5% и 3 мс.
- Сохранить прежний `Serialize*` как независимый путь записи в destination; не реализовывать сразу оба пути через новый stream.
- Общие builders заголовков сохраняют зарезервированные нули: outer112B, compact80B, verifier96B. Независимые fixtures проверяют поля/порядок, чтобы общий builder не делал сравнение двух путей самопроверкой.
- Scratch объявляется раньше `SnapshotTransfer`: при ошибке stream синхронизируется до освобождения host destination. Sink не удерживает span после возврата.
- Ограничение callback span и ограничение дополнительной C++ heap-памяти проверяются отдельно. Test observer с заранее выделенным ledger учитывает peak одновременно живых C++ allocations в потоке вызова: не больше 16 MiB + 4096 B постоянного metadata overhead. Запрет одной большой allocation сам по себе этого не доказывает. Служебные C++ new/delete HIP в этом же потоке также учитываются; прямые malloc/device/OS allocations и другие потоки этим observer не охвачены.
- Legacy fallback неизвестного draft provider сохраняет full allocation; строгий bounded write-path квалифицируется для нашего Qwen+MTP. DFlash/auto не получают нового обещания памяти.
- Рабочий main возвращается после каждого GPU-окна. Старые binaries/raw/FAIL сохраняются. Push, перевод, установка pi и принятие Q1 не выполняются.

## Интерфейсы и файлы

1. Новый `src/core/snapshot_stream.hpp`: `gufo::core::PersistentSnapshotSink = std::function<void(std::span<const std::uint8_t>)>`, `kSnapshotStreamChunkBytes = 16 * 1024 * 1024`, helper выдачи borrowed host span порциями без копии. `TextModelRunner::SnapshotSink` остаётся совместимым alias.
2. `IDraftBackendSnapshot::StreamPersistent(const core::PersistentSnapshotSink&) const` в `src/core/speculative/draft_backend.hpp`: совместимый default через старый serializer с проверкой размера. `QwenMtpDraftSnapshot` в `src/models/qwen/hip/mtp/backend.cpp` переопределяет его и выдаёт свой host payload без полной копии.
3. `SpeculativeVerifierSnapshot::StreamPersistent(const core::PersistentSnapshotSink&) const` в `speculative_verifier.hpp/.cpp`: прежние header, little-endian rolling float bits и draft stream; контроль суммарного количества байтов и исключений.
4. `QwenGpuSnapshot::StreamCompact(const core::PersistentSnapshotSink&) const` в `src/models/qwen/hip/executor.hpp` / `arena.cpp`: header → K → V → conv → DeltaNet → rope. Snapshot planes уже packed; F16 token-major и F32 head-major выдаются без transpose. Один scratch переиспользуется.
5. `QwenTextRunner::StreamPersistentSnapshot` в `src/cli/serve/inference_backend.cpp`: outer header → target → logits → verifier, прежние validation и размеры. Restore и disk store не перестраиваются.
6. Узкая внутренняя фабрика `CreateQwenTextRunner(model, options)` в новом `src/cli/serve/qwen_text_model_runner.hpp` и существующем backend.cpp. Options содержит прежние max_context, dflash/mtp shared models, SpeculativeOptions и artifact fingerprints. Production loader и тест используют одну фабрику; загрузка моделей/новые CLI options не добавляются.

## Шаг 1. Настоящий RED до streaming override

Файлы: новая фабрика, `tests/models/qwen/persistent_snapshot_stream_test.cpp`, регистрация рядом с существующим `qwen_serving_test_support` в `tests/models/qwen27b/CMakeLists.txt`.

- [x] Создать bounded synthetic Qwen model через существующие synthetic weights, настоящий runner/state/snapshot с payload больше 16 MiB. Setup и reference/output allocations — до наблюдаемого участка.
- [x] Test-only thread-local allocation guard запрещает отдельную C++ allocation больше 16 MiB только внутри настоящего runner.StreamPersistentSnapshot и ведёт preallocated pointer/size ledger для peak-live limit 16 MiB + 4096 B. Observer не выделяет память рекурсивно; исчерпание его фиксированного ledger — ошибка, не потеря учёта. Reference/output storage выделены заранее, callback пишет в заранее выделенный destination. Старый полный vector должен дать наблюдаемый allocation RED, а не ошибку конфигурации или skip. Несколько накопленных малых chunks тоже должны быть отвергнуты guard.
- [x] RED фиксирует requested bytes отвергнутой allocation, равные computed persistent payload. Observer учитывает scalar/array/aligned new/delete. Заморозить исходники/patch теста и выполнить отдельный Linux synthetic GPU RED с точным baseline, устойчивым запуском и возвратом main. Production streaming ещё не исправлен в этом baseline.

**Фактический результат:** RED3 подтверждён05:08 MSK, source b641 + patch f59f6aa7 / binary ae6fdad9; два setup FAIL сохранены отдельно. Это ещё не GREEN.

## Шаг 2. Реализация и узкие GREEN

Дополнительные тесты: `tests/core/test_speculative_verification.cpp`, `tests/models/qwen/hip/snapshot_streaming_test.cpp`, существующий disk-store тест. Нижний GPU target регистрируется в `tests/models/qwen/CMakeLists.txt`.

- [x] Реализовать перечисленные stream методы и общие header builders; старые direct Serialize-пути сохранить.
- [x] В synthetic verifier проверить actual streaming provider, который отвергает вызов legacy Serialize; затем legacy fallback, ошибочный byte count, overflow и бросающий sink. Linux target `speculative_verification_test` HIP-gated в source/CMake, но эти fixtures не запускают модель.
- [x] GPU fixture: F16/F32 KV, FP32/BF16 recurrent state, пустой/частичный prefix и область больше cap с хвостом, несколько layers/heads, rope version2. Stream равен старым bytes, восстановленные seeded значения проверяются независимо.
- [x] Повторить настоящий runner RED как GREEN: allocation guard, callback cap, byte count/equality, throwing sink после GPU chunk и внутри следующей области, повторная успешная сериализация неизменного snapshot.
- [x] Проверить, что частичная выдача + исключение не публикует новый disk file и сохраняет прежний checkpoint; использовать существующий store target.
- [x] Проверить actual MTP/pending branch: drafts > 0, размер самого MTP payload больше 16 MiB и отдельный asserted pending_count > 0, byte-exact snapshot/reload. Это делает guard чувствительным к скрытой полной копии MTP. Text-only case без pending не засчитывается за эту ветвь; при необходимости отдельный узкий real-model case сохраняет собственный scope. Vision grids проверяются отдельно.
- [x] Общий required format-check с сохранением прежних diagnostics, затронутые tests и независимое ревью. Только после GREEN — логический локальный commit с CONTEXT/CHANGELOG.

**Фактический результат:** GREEN3 все три GPU tests PASS, CPU4/4 и format727 PASS, source36e46+patch1e492388; полный архивc60f6609. Зафиксировать логический commit, затем ordinary измерения.

## Шаг 3. Проверка эффекта на стенде

- [x] Собрать новый ordinary candidate тем же фактическим toolchain; сохранить старый b641/ef7e. Закрепить source/binary/harness/runtime и текущие библиотечные зависимости; не повторять ошибку сравнения разных окружений.
- [ ] Сравнить затронутые Save/Restore/C4 запросы с b641 при одинаковой работе и входах; сохранять каждую timing cell, несопоставимые истории и ошибки. При regression расследовать причину, не усреднять и не расширять допуск.
- [ ] Выполнить обязательную композицию после подготовительных проверок: те же четыре длинные истории, refill, отмены, изображения, полный час после warmup и исходный first/last20 memory gate. Source нового кандидата требует отдельного reviewer/manifest; прежний final3 не переименовывается.
- [ ] Независимо проверить raw, memory, kernel и возврат main. Обновить registry, candidate manifest, CONTEXT, оба журнала и эксплуатационные документы. Даже PASS этого исправления не принимает Q1 и не разрешает deployment.

## Риски для ревью

Неправильный порядок plane/rope bytes; освобождение scratch при незавершённом HIP copy; полный allocation, скрытый за нарезкой callbacks; непокрытый pending/MTP branch; рост latency от большего числа HIP synchronization. Для каждого предусмотрены отдельные проверки выше. Реальный RSS и скорость оцениваются измерениями, а не только размером scratch.

### Уточнение по измеренному отказу GREEN2

Scratch выбран8MiB при прежнем потолке16MiB и live guard16MiB+4096. GREEN1/2 подтвердили, что16MiB scratch вместе со служебными C++ allocations HIP превышает aggregate-live bound. Уменьшение scratch не меняет bytes/format/очерёдность/lifetime; разрешённый потолок не повышен. Влияние дополнительных Copy/sync обязательно проверить ordinary matched A/B. GPU GREEN3 подтверждён; влияние на latency/RSS всё ещё требует ordinary измерений.
