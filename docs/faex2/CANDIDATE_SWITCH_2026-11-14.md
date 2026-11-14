# Проект переключения кандидата b641fe3e

**Статус: проект для проверки; комплект launcher не сгенерирован, перевод не
разрешён и не выполнен.** В [manifest](CANDIDATE_2026-11-14.json) установлено
`ready_for_deployment=false`. Ordinary `final3` завершён 2026-11-14 в 04:08 MSK: **NOT_ACCEPTED / FAIL_MEMORY** по RSS (+2,34 GiB при пороге 1 GiB). Полные raw и strict review сохранены; остальные восемь gates PASS. Main1195/e319 восстановлен и независимо проверен READY PID915853. Продолжается разбор host-памяти; этот проект не разрешает переключение.

Документы [кандидата 13 ноября](CANDIDATE_2026-11-13.json) и
[его процедуры](CANDIDATE_SWITCH_2026-11-13.md) остаются историческими.
Согласия на другие сборки и подготовку этого проекта не разрешают перевод.
Ниже описано будущее окно после закрытия gates и отдельного одобрения
владельцем точного кандидата и проверенного комплекта возврата.

## Артефакты и границы доказательств

| Назначение | Source SHA | SHA-256 бинарника |
|---|---|---|
| Обычный кандидат | `b641fe3eb4d6a91352a007587b4f17a440fc4aff` | `ef7e3040138761614641311e12efe6e335dfbd5e5b244b578a0e4b4833310cb1` |
| Сохранённый рабочий / будущий recovery | `1195d15427a43c45516ab9e7dd90777f8216b326` | `e319b5f5b5198dc60e089d8b57b3e8826728aed5b6ba9f1534adf0b18a67d607` |
| Только диагностика cache5 | `b641fe3eb4d6a91352a007587b4f17a440fc4aff` | `5b80416539494bb64574092ae305a442564fd26935087c5684fb0c22f164eb9f` |

Обычная Linux-сборка завершена в 02:30:26 MSK. Три затронутых CPU-targets
прошли: `continuation_cache_test`, `continuation_disk_store_test`,
`text_model_runner_test`. Source/toolchain и файлы закреплены в
[build manifest](../../.local/remed-20261113/codex-closeout/remote-evidence/build-minstep1/cand/b641fe3eb4d6a91352a007587b4f17a440fc4aff/MANIFEST.json).
Изменение — ранняя advisory-проверка `min_step` перед удержанием snapshot
очередью. Точная RAM freeze/publication и финальная проверка writer сохранены;
из-за изменения момента решения одинаковые будущие cache hits не обещаются.

Пример этого различия: durable prefix A ещё существует, пока writer сохраняет
посторонний checkpoint C. Новый checkpoint B продолжает A менее чем на 2048
токенов и пропускается ранней проверкой. Затем публикация C вытесняет A по LRU.
Прежний writer мог бы сохранить стоявший за C checkpoint B, уже не обнаружив A;
после раннего пропуска B в очереди нет. Если подходящая RAM-копия тоже потеряна,
будущее продолжение может потребовать полного prefill. Это проверенный CPU-тестом
компромисс политики, а не установленная причина двух старых misses final2.

| Проверка | Состояние и оставшееся обязательство |
|---|---|
| Ordinary final3 | **NOT_ACCEPTED / FAIL_MEMORY**: RSS first/last max6628700160→9136889856, delta2508189696 bytes; остальные8gates PASS. 3610,648с,1610requests/768long/192C4/192cancel/4postrefill. Полный raw проверен. Тот же harness `76bd2ef2`, nonce `6af771b5`, два calibration, четыре fill, warmup и не менее 3600 с после него, без agent holds. Memory gate первых/последних 20 минут прежний: 1200 с с каждого края, минимум 20 samples, FAIL при росте максимума или медианы выше 1 GiB |
| Cache5 | 14 запросов: 13 точных полных ответов и одна подтверждённая отмена; target disk 128309 и live 117825, четыре physical width4 с drafts. Окно и возврат прошли, но focused memory — `INCONCLUSIVE_MONITOR`: gap 310.258 мс > заранее заданных 250 мс. Все 24 timing flags сохранены; comparative timing — `INCONCLUSIVE` |
| Предыдущая композиция | `final2` на `6d52ac57/27ef2c9c` остаётся `FAIL_MEMORY/NOT_ACCEPTED`. Две полные перечитки остаются OPEN. Новые cold rereads, partial C4 и тайминги final3 надо разобрать по каждому запросу; квота допустимых misses не вводится |
| Скорость и качество | Прежние 3 REGRESSION speed1 и последующие INCONCLUSIVE не закрыты. Пороги одновременно 5% и 3 мс для каждой фазы/ранга не меняются. Численные и cold-vision FAIL на `feae3fc6/d4df0f68` сохранены; [Q1](PREFILL_QUALITY_CONTRACT_DRAFT.md) — неутверждённый проект |
| Агенты | OpenCode PASS относится к `6d52ac57/27ef2c9c` и `1195d154/e319b5f5`; на b641 не повторён. `pi` не установлен и не проверен. Это inherited evidence, не новый agent PASS |
| Перевод и recovery | Новый launcher не подготовлен и не установлен. Нужны проверка точного комплекта, его backup/recovery и отдельное одобрение владельца |

В cache5 непосредственно подтверждено: PINE не удерживается очередью и
освобождается при следующем RAM eviction; MOON также не enqueued, но остаётся
в RAM; LAKE enqueued и отклонён поздней проверкой writer. Причина раннего
fall-through у LAKE — busy либо ещё не подходящий durable prefix — по trace
не различена. Порядок Acquire отличается от cache4; всё снижение sampled peak
нельзя приписать только новому check. См. [lifetime review](../../.local/remed-20261113/codex-closeout/cache5-strict-manual-review.json)
и [raw/timing review](../../.local/remed-20261113/codex-closeout/cache5-independent-workload-review.md).

Отдельный [runtime audit](../../.local/remed-20261113/codex-closeout/composition-runtime-provenance-audit.json)
SHA `c063052176cf5421d9ec97a94b53def82ea40582aac6fb8555d93692a2438572`
зафиксировал различия окружения:

| Окно | Python клиента | OpenSSL SONAME в candidate `ldd` |
|---|---|---|
| final2, ordinary 6d52 | 3.14.9 | `libssl.so.3`, `libcrypto.so.3` |
| final3, partial ordinary b641 | 3.14.10 | `libssl.so.4`, `libcrypto.so.4` |
| cache4, diagnostic 6d52 | 3.14.10 | `libssl.so.4`, **`libcrypto.so.3` и `libcrypto.so.4`** |
| cache5, diagnostic b641 | 3.14.10 | `libssl.so.4`, `libcrypto.so.4` |

Строка kernel и зарегистрированные переменные окружения совпали, как и версии
cc/HIP у всех четырёх сборок. Это не означает одинаковый runtime: ни final2/final3,
ни focused-пару cache4/cache5 нельзя считать сравнением в идентичном окружении
для квалификации производительности. Позднейшая проверка журнала пакетов
подтвердила обновления 01:34:46–54 MSK, включая Python/OpenSSL/curl и системные
LLVM/CMake; кто их запустил, не установлено. [Выдержка журнала](../../.local/remed-20261113/codex-closeout/runtime-package-history.json)
имеет SHA `4b74b828c519085eedf667d3397d6b5844eb34f5c9ceaa83efba50c5d30ee11f`.
Это объясняет смену версий, но не устанавливает причину конкретных пауз. Прямое наблюдение
ownership в cache5 сохраняется; `INCONCLUSIVE_MONITOR` и 24 timing flags не
переквалифицируются. Самостоятельные correctness и hour gates внутри final3
остаются применимы: final3 завершён отказом по RSS, остальные восемь gates прошли.

План final3 [закреплён](../../.local/remed-20261113/codex-closeout/final3-proposal-b641/plan-final3.json)
SHA `1a320db88d7acfcfe482b451eaf9b98099c2c00ef15020b98028e5b4bb149001`,
[конфигурация](../../.local/remed-20261113/codex-closeout/final3-proposal-b641/config-final3.json)
SHA `a1592a4df69e337f1b03d965b82fc18acb7308f4d502821e356aee89b846053a`.
Предварительная [проверка входов](../../.local/remed-20261113/codex-closeout/final3-proposal-b641/input-provenance-proof.json)
не заменяет результат исполнения. Финальный анализ должен использовать
[новый reviewer](../../.local/remed-20261113/codex-closeout/final3-proposal-b641/final3-review.py)
и [заранее заданный ledger](../../.local/remed-20261113/codex-closeout/final3-proposal-b641/analysis-ledger-preregistered.json),
сохраняя все raw, ошибки, misses и флаги.

## Подготовка комплекта после завершения приёмки

Сначала прочитать [локальную памятку](../../.local/HANDOFF.md), затем в
эксплуатационном проекте `LLM_FAEX2` — `AGENTS.md`, текущий
`PROJECT_CONTEXT.md` §§0/0.1, последние записи `CHANGELOG.md` и
`docs/gufo-faex2-runbook.md`. При отсутствии актуальных данных исполнение
останавливается; доступы, bind, роли и пути не угадываются.

Используется прежний `tools/gufo_make_cutover.py`. Источники подтверждённой
процедуры: `results/2026-11-12-gufo-cutover-v13/{README.md,cutover-info.json}`,
там же `prepare.sh`, `cutover.sh`, `recovery-check.sh`, `verify.py`,
`disk_check.py`, `launcher-v13/`; установленные снимки находятся в
`configs/home/truezero/bin/`. Это ссылки внутри эксплуатационного проекта,
не новый созданный комплект. Если фактический launcher уже не v13, проект
требует пересмотра от реального состояния.

1. Сверить SHA и `--version` обоих бинарников, фактические модель/mmproj,
   TheRock и окружение с manifest. Зафиксировать owner/child identities,
   launcher request, bind, cache namespaces и отсутствие другого окна модели.
2. Сохранить три установленных файла `faex2_owner.py`, `faex2-launch.json`,
   `fork_restart.sh` в эксплуатационный проект и отдельный серверный backup
   с `SHA256SUMS`. Бинарник `1195d154-e319b5f5` и прежние каталоги состояния
   сохраняются без пересборки и удаления.
3. Подготовить новый комплект существующим генератором. Ниже проект команды
   из корня эксплуатационного проекта; в рамках подготовки этого документа
   она **не выполнялась**. Номер 14 применим только после сверки baseline v13:

   ```bash
   python3 tools/gufo_make_cutover.py \
     --version 14 \
     --source-sha b641fe3eb4d6a91352a007587b4f17a440fc4aff \
     --binary-sha ef7e3040138761614641311e12efe6e335dfbd5e5b244b578a0e4b4833310cb1 \
     --what 'исправления аудита и ранний min_step перед удержанием snapshot очередью' \
     --built-at '$HOME/gufo-remed-20261113/cand/b641fe3eb4d6a91352a007587b4f17a440fc4aff/gufo' \
     --acceptance results/2026-11-13-gufo-remediation \
     --out results/2026-11-14-gufo-cutover-b641fe3e \
     --stand-dir gufo-cutover-20261114-b641 \
     --prev-out results/2026-11-12-gufo-cutover-v13
   ```

4. Проверить generated diff, `cutover-info.json`, все SHA, namespaces,
   роли main=b641 и recovery=1195, неизменность wrapper и отсутствие
   экспериментальных overrides. Выполнить CPU-тесты launcher с сохранением
   настоящего exit code. Проверить сценарии вручную. После проверки
   `prepare.sh` должен подтвердить SHA закреплённой копии и backup;
   `PREPARE_OK` не означает перевод. Представить конкретный комплект владельцу.

## Одобренное окно, проверки и возврат

Только после отдельного одобрения выполнить проверенный generated
`cutover.sh` в устойчивом к разрыву SSH сеансе. Старый процесс останавливается
старым launcher; новый комплект устанавливается после подтверждения остановки.
Профиль остаётся `fork_restart.sh mtp1 4 131072 mmproj`, draft 1, reasoning 512,
RAM 16 GiB/disk 64 GiB. Сеть, порт, доступы, ОС и модель не меняются.

После READY сверить полный SHA, `/proc` executable, identities, launcher request,
`/health`, `/v1/models`, `/slots` и bind. Проверить ответ 56, реальный `get_weather`
для Казани, синтетический красный квадрат, SSE/JSON без framing leak, реальные
MTP proposed/accepted и независимость сеансов; сохранить raw. Исторические
`verify.py`/`disk_check.py` требуют проверки содержимого: их exit 0 сам по себе
не означает PASS. Поздняя ошибка API, identity или disk restore требует возврата,
даже если автоматическая защита старта уже сочла READY успешным.

Для `qwen-bf16-fp32-prefill-v2` нужен отдельный namespace. Это численная identity,
не версия snapshot file format. Старое несовместимое состояние должно дать
miss; переименовывать его identity нельзя. На кандидате создать собственное
состояние, дождаться durable store, перезапустить тот же бинарник и подтвердить
disk hit, границу snapshot, токены и ответ. Для raw zero-prefill нужен ровно 0;
остаточный chat prefill не называть нулевым. Guard NPU/внешнего hipBLASLt plan
без content fingerprint не обходить.

Проверить настоящий recovery: остановить candidate, запустить сохранённый
1195 через новую проверенную роль recovery, проверить SHA/READY/API и остановить
его. Поддерживаемый селектор после установки нового launcher:

```bash
env GUFO_BIN="$HOME/gufo-release/1195d154-e319b5f5/bin/gufo" \
  "$HOME/bin/fork_restart.sh" mtp1 4 131072 mmproj
```

Статус и остановка recovery используют тот же `GUFO_BIN` и существующие
`--status`/`--stop`; нового флага `--role` нет. Возвращать candidate как main
можно только при успешном recovery и сохранении всех одобренных gates, затем
проверки повторяются. Если неисправен launcher, остановить только принадлежащие
ему процессы, восстановить три файла из backup, сверить SHA и вернуть прежний
main. Не останавливать процессы по имени, не пересобирать бинарник и не удалять
кэши. Завершить окно только после подтверждённого рабочего сервера.

Сохранить простой, статусы, SHA, backup, конфигурацию и kernel/server logs;
обновить эксплуатационные документы и журнал форка. Готовность и допуск меняются
по новым доказательствам и решению владельца. Прежние FAIL остаются в истории.
