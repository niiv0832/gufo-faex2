# Итоговая таблица доработок аудита — 2026-11-14

Таблица перенесена из проверенной ACTUAL v4; изменены только заголовок и относительные пути ссылок. Исходный SHA-256: `4d53bd0c18961c4dc2ee5a3a7ee207c609b215174cc59bb33a51e304ff25352d`. Ссылки в `.local/` указывают на непубликуемые доказательства владельца.

**Final4: VERIFIED, все 9 зарегистрированных composition/restore gates PASS.**
Это scoped hour PASS, **общая приёмка остаётся NOT_QUALIFIED**:
comparative timing INCONCLUSIVE, два full JSON misses final2 OPEN/NOT_PROVEN,
Q1 DRAFT/NOT_APPROVED, pi NOT_EXERCISED, новый candidate agent replay отсутствует.
`ready_for_deployment=false`; новый launcher не установлен, deployment не разрешён.

Root strict review [F4] проверил 1600 rows:1409 полных ответов,191 подтверждённая
отмена,764 long continuations,191 полных C4 и4 post-refill проверки. Нагрузка
после warmup3602,261с. Эти191cycles включают1warmup +190load cycles; число
запросов post-warmup1587. Новых full long JSON misses и partial C4 не обнаружено.
Прежние final1/final2/final3 FAIL не переквалифицированы.

Источник — [неизменённая v3](../../.local/remed-20261113/codex-closeout/AUDIT_CLOSURE_TABLE_DRAFT_V3.md) и уже сохранённые
reviews/raw. Эта карта агрегирует доказательства; suites/SSH/API/GPU не
повторялись. Все14 исходных,6 повторных и7 независимых пунктов перечислены ниже.
Исторические model/agent результаты feae/6d/1195 не приписываются ELF8acf.
GREEN3 — отдельные test binaries,16changed source/test SHA совпали с ffc;
actual final4 и B2 — отдельные исполнения ordinary8acf.

## Первоначальные №1–14

| № / предмет | Текущий статус | Исправление / выполненная работа | Точное сохранённое доказательство | Остаток / граница |
|---|---|---|---|---|
| 1. Target MTP против scalar AR | ИСПРАВЛЕНО В ПРОВЕРЕННОМ SCOPE | `59ff3f5f`, затем `97e185d5`: alpha/beta row path использует тот же fused kernel для всех форматов | [AR/verify](../../.local/remed-20261113/results/final-tests/model-qwen_ar_verify_parity_test.out): 11 сценариев PASS; [rows ops](../../.local/remed-20261113/results/final-tests/model-qwen_decode_rows_ops_test.out); [FT] | Результаты feae, не универсальная exact-квалификация всех моделей/режимов/длин; comparative timing отдельно OPEN |
| 2. Продолжение после rollback | ИСПРАВЛЕНО В ПРОВЕРЕННОМ SCOPE | Следствие исправления №1; отказ/откат/фиксация и дальнейшее продолжение проверены | Тот же [AR/verify](../../.local/remed-20261113/results/final-tests/model-qwen_ar_verify_parity_test.out): после отказа и фиксации следующие токены совпадают; [FT] | Не расширять утверждение за покрытые сценарии |
| 3. Поздний ShareCheckpoint | ИСПРАВЛЕНО; HTTP-ПРОВЕРЕНО | `4550e3b9`: не вставлять границу внутрь уже выполняемого прохода | `TestLayeredPrefillKeepsCheckpointAfterSharedArrival`, [CPU scheduler](../../.local/remed-20261113/results/final-tests/cpu-text_generation_scheduler_test.out); [G2-before]: late_shared_prefix и answers PASS | Цена пути остаётся в сравнительном timing scope: follower может самостоятельно дочитать prefix; это не ожидание/restore бесплатно |
| 4. DeviceInput RestoreLayout exception safety | ИСПРАВЛЕНО НА УРОВНЕ DeviceInput | `2fa360e9`: подготовка нового layout до освобождения старого/публикации | [GPU device-input test](../../.local/remed-20261113/results/final-tests/gpu-qwen_vision_device_input_test.out), [FT]; исходный RED: 6 отказов →0 в REMEDIATION | Настоящий backend с injected device-write failure не проверен; реальная порча/OOM на стенде не заявлялись |
| 5. Частичный AllocateSsmReplayLog | ИСПРАВЛЕНО НА УРОВНЕ ARENA | `d81c883e`: готовность полного набора четырёх буферов | [GPU replay allocation](../../.local/remed-20261113/results/final-tests/gpu-qwen_ssm_replay_allocation_test.out), [FT]; RED6→0 в реестре | Моделируемая ошибка выделения не означает наблюдавшийся production OOM |
| 6. ScoreTogether / EnsurePrimeScratch | ИСПРАВЛЕНО | `ae550793`: неполный набор освобождается, повтор инициализации безопасен | [Model shared-score memory](../../.local/remed-20261113/results/final-tests/model-qwen_mtp_shared_score_memory_test.out), [FT]: retry и balance allocation/free | Сохраняются границы fault-injection; общий расход памяти — №7 |
| 7. Учёт и расход памяти | УЧЁТ/WRITE-BUFFER ИСПРАВЛЕНЫ; SCOPED HOUR PASS | `46c1e59d`: retained/reserved; `b641fe3e`: early min_step; `ffc8580a`: exact Qwen/MTP stream8MiB scratch | [ECPU], [C4], [C5], [HOST], [GREEN3]; [F4]:memory gate PASS по прежнему1GiB/first-last1200s,239samples на край; [BUILD-FFC] | Старые [F2]/[F3] FAIL_MEMORY сохранены. Это один hour gate, не общий RSS/HIP bound, не24/7 и не доказательство единственной причины старого пика; generic fallback допускает full-vector |
| 8. Численная identity disk cache | ИСПРАВЛЕНО В ЗАЯВЛЕННОМ GPU SCOPE | `6286db7c`, `70a8b07a`, `9738fd03`; `6d52ac57` fail-closed для внешней арифметики без fingerprint | [Cache policy](../../.local/remed-20261113/results/final-tests/model-cache-policy.out), [FT]; [ID-RED]/[ID-GREEN], Linux [ID-LINUX] | NPU/внешний plan persistence не поддержаны без content identity. Guard покрывает также ручной fingerprint с disk disabled. Не закрывает два misses; см. А/I3 |
| 9. Приближённый short/grouped prefill | OPEN / QUALITY NOT_ACCEPTED | Операторное разделение `7e9a70ce`, scalar logits/ΔNLL tools, независимый requant oracle `2acb8673`, meas4 corpus | [MEAS4], [REQUANT-RED]/[REQUANT-GREEN], [Rows](../../.local/remed-20261113/results/final-tests/model-qwen_moe_prefill_rows_test.out) | Обнаруженные положительные short ΔNLL сохранены. Q1 DRAFT, margins не приняты; нового holdout/evaluator нет. Grouped-off тоже approximate; lossless не заявлен |
| 10. BF16 router fallback | ИСПРАВЛЕНО; SYNTHETIC ПОКРЫТИЕ | `80d3a1cb`: BF16 не читается как F32 | [Router fallback ops](../../.local/remed-20261113/results/final-tests/gpu-qwen_moe_router_fallback_ops_test.out), [FT]; исходная ошибка≈21→≈10⁻⁶ | Закреплённый GGUF имеет F32 router; synthetic BF16 не выдаётся за его runtime path; см. Е |
| 11. Sampled MTP oracle/chain/distribution | ПРОБЕЛ ТЕСТОВ ЗАКРЫТ В SCOPE | actual-q oracle/residual/chain; sampling race исправлена; поздние `9014f41c`/`de6bd584` усилили реальные ветви | [Chain](../../.local/remed-20261113/results/final-tests/model-qwen_sampled_mtp_chain_test.out):26841 проверка,18 residual,28 полных шагов; [FT]:4 контекста×10000, rc0 | Статистическое неотклонение не доказывает равенство всех распределений. MTP1; EOS полного шага plain, маска verifier; глубина>1 и непроверенные сочетания не покрыты |
| 12. Vision embedded MTP35B | ТЕСТ ИСПРАВЛЕН; COLD PARITY FAIL | `7bd04f4f`: actual embedded MTP/размер projector; exact snapshot restore исследован отдельно | [Vision serving](../../.local/remed-20261113/results/final-tests/model-vision-serving-mtp.out): rc1 сохранён; CLOSEOUT «Зрение», [FT] | Exact restore PASS не заменяет cold-prefill parity. Падающий cold/scalar путь не удалён, относится к №9/Q1 |
| 13. API и реальные агенты | API/OPENCODE В ИСТОРИЧЕСКОМ SCOPE; NEW ORDINARY COMPOSE PASS; PI NOT_EXERCISED | Изоляция/MCP/finalized/SSE/empty-pass guards; Btrfs bridge; B2+final4 настоящие ordinary requests | [G2-after], [G2-before], [OC], [H44], [HREPLAY], [BRIDGE]; [B2], [F4] отдельно | Formula INHERITED_FAIL сохранён. OpenCode на8acf не повторён; Pi не установлен/не исполнен. HTTP composition не заменяет отдельный agent workflow или полный API/vision sweep |
| 14. Итоговая композиция | FINAL4 VERIFIED:9/9GATES PASS; HISTORICAL FAIL СОХРАНЕНЫ | `76bd2ef2` refill history; exact same plan/nonce/3600s afterwarmup/first-last20 gate исполнен наffc | [F4]:3602,261с/1600rows/764long/191C4/191cancel/4postrefill; [F4-RESTORE]:main970495 independently READY08:11:18, exact identity/SHAe319, kernelCLEAN | Не качество/скорость/24/7/deployment. Два старых full JSON misses OPEN; отсутствие новых вfinal3/final4 не устанавливает их причину. Independent peer status указан отдельно ниже |

## Повторное ревью А–Е

| № / предмет | Текущий статус | Исправление | Доказательство | Остаток / граница |
|---|---|---|---|---|
| А. Старая numerics-v1 принимала новые snapshots | ИСПРАВЛЕНО | `70a8b07a`: v2 + явные default policies | REMEDIATION А: old6685/6692 hit → новый miss6692 → новый own restart hit6685; [Cache policy](../../.local/remed-20261113/results/final-tests/model-cache-policy.out), exact after_snapshot_restore в [AR/verify](../../.local/remed-20261113/results/final-tests/model-qwen_ar_verify_parity_test.out) | Старый numeric namespace несовместим; новая проверка не даёт права мигрировать/переименовать старую identity; №8/I3 сохраняются |
| Б. PASS после аварии/пустых gates | ИСПРАВЛЕНО | `a24a6756`: missing/malformed restart state, exception и невыполненный этап дают FAIL | REMEDIATION Б с RED→GREEN; [G2-before]/[G2-after] сохраняют действительные статусы, [H44] дополняет поздние guards | Не заменять KNOWN_FAILURES общим PASS; поздние агентские false PASS закрыты отдельно ниже |
| В. Непроверенные ответы соседей при cancel | ИСПРАВЛЕНО | `a24a6756`: полный expected ответ каждого соседа, обязательный native cancelled | REMEDIATION В: wrong products ранее PASS→FAIL; [G2-before] sessions_after_cancel PASS; [F2]179 native cancellations | Показанные сценарии; произвольная отмена/любая нагрузка не объявляются проверенными |
| Г. Пустой/неполный/overflow запуск parity tool | ИСПРАВЛЕНО | `74dd41dc`: строгие args/files/ID до загрузки модели | REMEDIATION Г: no files, third file, overflow отклонены rc2; [AR/verify](../../.local/remed-20261113/results/final-tests/model-qwen_ar_verify_parity_test.out)11 scenarios PASS | Исправлен CLI/test trust, не новое расширение model scope |
| Д. Oracle предполагал нормированное q | ИСПРАВЛЕНО | `72a75454` и последующие: declared q отделено от actual proposal law; residual независимо | [Oracle](../../.local/remed-20261113/results/final-tests/cpu-sampled_speculation_oracle_test.out), [Chain](../../.local/remed-20261113/results/final-tests/model-qwen_sampled_mtp_chain_test.out); CLOSEOUT sampled | Условные статистические границы/float effects сохраняются; не универсальный lossless |
| Е. Quantized router принимался как F32 | ИСПРАВЛЕНО ОТКАЗОМ ПРИ ЗАГРУЗКЕ | `355b1d86`: router/shared gate только F32/BF16 | [Router type](../../.local/remed-20261113/results/final-tests/cpu-qwen_moe_router_type_test.out), [FT] и реальная загрузка F32 GGUF | Quantized router не поддерживается; это fail-closed, не реализация других форматов |

## Независимое ревью 86c8fd93..ec8656b1

| ID / предмет | Текущий статус | Исправление / актуальное уточнение | Доказательство | Остаток / граница |
|---|---|---|---|---|
| I1. Broken output/signals/bench cleanup/monitor/VRAM | ИСПРАВЛЕНО В HARNESS | No-throw logging, owned process/session cleanup, signal handling; mandatory monitor и GTT+VRAM gate | [ORCH38]:38 tests PASS; [ORCH-SRC]: test_benchmark_server_left_by_the_driver_is_stopped, missing-metrics/VRAM/composition negatives | Историческое «остальное в работе» устарело. Успех оркестратора не означает PASS модели/памяти; исходные FAIL сохраняются |
| I2. PASS скорости при missing tables | ИСПРАВЛЕНО | Отсутствующие/пустые measurements не дают PASS | [ORCH-SRC]: test_missing_measurements_are_never_a_pass; [ORCH38] | Сам comparative speed gate OPEN/INCONCLUSIVE; исходные3 REGRESSION остаются |
| I3. NPU и plan не полностью идентифицировались | ИСПРАВЛЕНО FAIL-CLOSED | `9738fd03` оказался недостаточен для content; `6d52ac57` запрещает unsafe persistence | [ID-RED]/[ID-GREEN]:5 unsafe cases; [ID-LINUX]; [Commit review](../../.local/remed-20261113/codex-closeout/final-commit-review.md) | NPU/внешний plan не квалифицированы; GUFO_VISION_ATTENTION=tiled отдельно не проверялся, image encoder identity остаётся отдельной |
| I4. Alpha/beta проверены только F32 | ИСПРАВЛЕНО В ROW PATH | `97e185d5`: fused kernel при любом формате; тест требует оба типа слоёв и≥30 сравнений | [Rows ops](../../.local/remed-20261113/results/final-tests/model-qwen_decode_rows_ops_test.out), [AR/verify](../../.local/remed-20261113/results/final-tests/model-qwen_ar_verify_parity_test.out), CLOSEOUT | Packed27B rank48/hidden5120 не изменялся и не проверялся этим исправлением; не выдавать real35B за все модели |
| I5. Тесты residual/requant/пустых весов обещали лишнее | ИСПРАВЛЕНО В ПРОВЕРЯЮЩИХ СРЕДСТВАХ | Усилен actual-q/residual/chain; независимый requant decoder/bound; mandatory operator comparisons | [Chain](../../.local/remed-20261113/results/final-tests/model-qwen_sampled_mtp_chain_test.out), [REQUANT-RED]/[REQUANT-GREEN], [Rows ops](../../.local/remed-20261113/results/final-tests/model-qwen_decode_rows_ops_test.out) | Историческое «sampled/переквантование в работе» устарело. Это не устранение approximate prefill quality FAIL |
| I6. Учёт только target executor без полного server path | СЕРВЕРНЫЙ ПУТЬ ИЗМЕРЕН; SCOPED FINAL4 MEMORY PASS | Exact streaming убрал полный Qwen write-buffer при прежних format/identity/budgets; original hour gate исполнен | [F2], [C4], [C5], [F3], [HOST], [GREEN3], [F4] | Historical memory FAIL/cache5 INCONCLUSIVE не заменены. First-last максимумRSS+452956160B, GTT+291520512B, GPU+291737600B ниже1GiB. Это не вся RSS во всех сценариях и не comparative speed PASS |
| I7. Цена позднего shared checkpoint | ПОВЕДЕНИЕ УСТАНОВЛЕНО; TIMING OPEN | Если boundary уже внутри выполняемого pass, follower prefills сам, wait0 | [G2-before]: late_shared_prefix/answers; REMEDIATION независимое ревью; [SPEED], [D2] | Это допустимый описанный путь, но строгие per-request/phase5%+3ms не закрываются функциональным PASS или средним throughput |



## Фактический hour gate, без замены его правил

| Счётчик | Рост максимума, B | Рост медианы, B | Результат |
|---|---:|---:|---|
| rss_bytes | 452956160 | 52699136 | PASS |
| gtt_used_bytes | 291520512 | -94355456 | PASS |
| vram_used_bytes | 45056 | 180224 | PASS |
| gpu_used_bytes | 291737600 | -94208000 | PASS |

Каждый край —1200с и239samples; предел1073741824B. Рост максимума **или**
медианы выше предела дал бы FAIL. Счётчики DRM device-wide не превращаются
в атрибуцию одного allocation. Main1195/e319 — сохранённый рабочий бинарник и
цель возврата после тестов; это не переименование фактической launcher recovery
role. [F4-RESTORE] подтверждает возврат; [F4-ARCHIVE] фиксирует полный архив SHA
`ad50b20a516a2e2a19b5a97e00ce7e48325d280aec55b3862eb7669bf15b2f04`.
Архив здесь повторно не хешировался: используется сохранённый receipt.

## Дополнительные обязательства после v3

| Обязательство | Фактическое состояние | Граница |
|---|---|---|
| Actual allocation RED→GREEN | CLOSED IN TESTED SCOPE:RED3 requested19542208; GREEN3 plainpeak8409971/remain0, actualMTP17302021/heap0, pending1/draft1/reload/resume,12compact/rope cases | [GREEN3]; отдельные testELF, не общий RSS/HIP/качество |
| Atomic store при throwing stream | CLOSED: old published bytes иrestart сохранены; badstream не публикует checkpoint/temp | `f59521a1`, [DISK-ATOMIC], GREEN3CPU; cachepolicy не менялась |
| Прежние9format failures | CLOSED:36e46be3, общая проверка727файлов PASS | [GREEN3]; новый Nix-run не заявляется |
| Btrfs runtime identity helper | CLOSED IN HELPER SCOPE:CPU32+main no-request validation; actual A2/B2 captures сохранены | [BRIDGE], [A2], [B2]; A1pre-requestFAIL сохранён. Late first module load делает сравнительную квалификацию INCONCLUSIVE |
| A2/B2 correctness/storage | PASS_SCOPED:9+9 ответов, четыре stable cache-image пары по полному SHA | [A2], [B2], [PAIR-INTEGRITY], [PAIR-EQUALITY]; не другие длины/входы/vision/arithmetic |
| A2/B2 comparative timing | INCONCLUSIVE_EVIDENCE / NOT_QUALIFIED | [PAIR-LEDGER]:184cells,9flags,46issues; late mappings/observer I/O, cold monitor gaps, C4 admission/work differences. Native/usage queue пары не независимые события |
| Full final4 против final3 | DIAGNOSTIC_ONLY / INCONCLUSIVE_HISTORICAL_RUNTIME_NOT_FULLY_OBSERVED | [F4-TIMING], [F4-COMPARE]:1610→1600rows,1600 exact input pairs +10control-only;2768flags на928candidate requests, все1952longflags и200work-difference pairs сохранены. Это не2768независимых причин и не доказанная причинная регрессия streaming; peer review отдельно ниже |
| Новый ordinary часffc | VERIFIED:9gates PASS | [F4]; scoped composition/restore, не переоценка всех старых FAIL и не comparative speed PASS |
| Полные JSON missesfinal2 | OPEN / OLD_CAUSE_NOT_PROVEN | [LRU]; INFO не различаетbusy/LRU/retention/key.93–95с не объявлены нормой. Новые успешные истории не доказывают причину старых misses |
| Q1/approximate prefill/coldvision | DRAFT_NOT_APPROVED / QUALITY_NOT_ACCEPTED | [Q1], [MEAS4]; proposed margins не приняты; feasibility gate недостаточности длинныхсрезов сохранён; holdout/evaluator не исполнены |
| Pi/currentcandidate agent replay | NOT_EXERCISED | OldOpenCode4/4×2 inherited; pi не установлен. Hour HTTP workload не заменяет реальныйagentworkflow |
| Deployment/launcher | NOT_AUTHORIZED / NOT_GENERATED | Ordinaryffc не назначен рабочим; main1195/e319 сохранён. Нужны оставшиесяgates, точныйпакет и отдельноеодобрение |

## Независимые peer reviews

- `independent_raw_review`: **SCOPED_INDEPENDENT_REVIEW_COMPLETE**. [final4-independent-full-review.json](../../.local/remed-20261113/codex-closeout/final4-independent-full-review.json), SHA `3c60117086f0f968842acbff5ac69991843ce25d4efc7144d3cdc590119f1e05`.
  [Краткий вывод](../../.local/remed-20261113/codex-closeout/final4-independent-full-review.md), SHA `f1f6e408d3618c4852ad44572996b40fa69dd340d6a2f6c66086fa386100149a`.
- `independent_full_timing_review`: **INDEPENDENT_FROZEN_HOUR_MEMORY_SUBGATE_PASS**. [final4-independent-memory-and-timing-review.json](../../.local/remed-20261113/codex-closeout/final4-independent-memory-and-timing-review.json), SHA `910c884b743a73844614d0ccf685bee822fa7d37bffbf9240fe194e68460b593`.
  [Краткий вывод](../../.local/remed-20261113/codex-closeout/final4-independent-memory-and-timing-review.md), SHA `4e6a6dbaea8cb4df94f0ea41c377559f5750b3592d00e97291f55faf77ede4a7`.
  PASS относится к memory subgate; полнота timing ledger проверена, comparative timing остаётся INCONCLUSIVE/NOT_QUALIFIED.

Full final4-vs-final3 timing ledger уже получен; окончательная независимая
проверка имеет собственный статус выше. Старые speed flags и A2/B2
INCONCLUSIVE действуют независимо от её готовности. [F4-RUNTIME] имеет
CAPTURES_COMPLETE, но INCONCLUSIVE_ENDPOINT_DIFFERENCE: добавились1ELF и1nonELF,
environment differences0. Endpoint capture не равен continuous runtime qualification.

## Полнота покрытия и происхождение

[COVERAGE-V5] содержит44commits ec8656b1..a9a51cc7: прежние42entries сохранены,
добавлены только docsdf310f53/a9a51cc7. Это completed commit coverage, не
заявление о ревью незакоммиченных final4-doc edits root. Нового production
кода послеordinaryffc этим продолжением не вводилось. Старые v3MD/JSON
сохранены неизменными; эта карта не меняет tracked документы.

## Доказательства

[FT]: ../../.local/remed-20261113/results/final-tests/summary.txt
[G2-before]: ../../.local/remed-20261113/results/gates2/check-before/report.json
[G2-after]: ../../.local/remed-20261113/results/gates2/check-after/report.json
[MEAS4]: ../../.local/remed-20261113/prefill/meas4/report-final/report.md
[REQUANT-RED]: ../../.local/remed-20261113/prefill/meas4/requant_test.red.log
[REQUANT-GREEN]: ../../.local/remed-20261113/prefill/meas4/requant_test.green.log
[ID-RED]: ../../.local/remed-20261113/codex-closeout/numerical-identity-red.log
[ID-GREEN]: ../../.local/remed-20261113/codex-closeout/numerical-identity-green.log
[ID-LINUX]: ../../.local/remed-20261113/codex-closeout/remote-evidence/cand/6d52ac57224b4d91d23fcbaba6fe053a411d65dc/numerical-identity-test.log
[OC]: ../../.local/remed-20261113/codex-closeout/opencode-actual-review.md
[H44]: ../../.local/remed-20261113/codex-closeout/final-harness-green.log
[HREPLAY]: ../../.local/remed-20261113/codex-closeout/final-harness-offline-replay.json
[F2]: ../../.local/remed-20261113/codex-closeout/final2-acceptance-review.json
[F3-PLAN]: ../../.local/remed-20261113/codex-closeout/final3-proposal-b641/plan-final3.json
[ECPU]: ../../.local/remed-20261113/codex-closeout/early-min-step/validation.json
[C4]: ../../.local/remed-20261113/codex-closeout/cache4-strict-manual-review.json
[C5]: ../../.local/remed-20261113/codex-closeout/cache5-strict-manual-review.json
[ORCH38]: ../../.local/remed-20261113/accept/test-dry-run.log
[ORCH-SRC]: ../../.local/remed-20261113/accept/test_remed_window.py
[SPEED]: ../../.local/remed-20261113/codex-closeout/acceptance-speed3-review.md
[D2]: ../../.local/remed-20261113/codex-closeout/diagnostic2-acceptance-review.md
[LRU]: ../../.local/remed-20261113/codex-closeout/cache-lru-policy-observability.md
[WORK]: ../../.local/remed-20261113/codex-closeout/final3-first20-cache-work-source-review.md
[RUNTIME]: ../../.local/remed-20261113/codex-closeout/composition-runtime-provenance-audit.json
[PACKAGES]: ../../.local/remed-20261113/codex-closeout/runtime-package-history.json
[Q1]: PREFILL_QUALITY_CONTRACT_DRAFT.md
[NOV14]: CANDIDATE_2026-11-14.json
[F3]: ../../.local/remed-20261113/codex-closeout/final3-b641-acceptance-review.json
[F3-MEM]: ../../.local/remed-20261113/codex-closeout/FINAL3_FULL_MEMORY_REVIEW.md
[F3-TIMING]: ../../.local/remed-20261113/codex-closeout/final3-full-all-role-timing-v2.json
[HOST]: ../../.local/remed-20261113/codex-closeout/FINAL3_EXISTING_HOST_BUFFER_TRACE_ADDENDUM.md
[BASELINE-RISK]: ../../.local/remed-20261113/codex-closeout/FINAL3_FIRST20_STREAMING_BASELINE_RISK.md
[STREAM-PLAN]: SNAPSHOT_STREAMING_PLAN.md
[RED1]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-red/remote-evidence/accept-out/codex-stream-red1/red-proof/verdict.json
[RED2-PREP]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-red-v2/manifest-v2.json
[MTP-PLAN]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-red/mtp-fixture-review-v2.md
[DISK-ATOMIC]: ../../.local/remed-20261113/codex-closeout/early-min-step/streaming-exception-continuation_disk_store_test.log
[FORMAT-STREAM]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-red/full-format-red1.log
[B641-REVIEW]: ../../.local/remed-20261113/codex-closeout/cache5-early-minstep-source-review.md

[GREEN3]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-green3-independent-review.json
[BUILD-FFC]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/staged-remote-v2/cand/ffc8580a6edb0eae5e26c2a99c0712084faa948d/MANIFEST.json
[BRIDGE]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/runtime-btrfs-v1/live-validation-independent-review.json
[COMPARE-V3]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-compare-v3-independent-source-review.json
[HOUR-FFC]: ../../.local/remed-20261113/codex-closeout/final4-proposal-ffc/proposal-freeze.json

[F4]: ../../.local/remed-20261113/codex-closeout/final4-ffc-acceptance-review.json
[F4-RESTORE]: ../../.local/remed-20261113/codex-closeout/remote-evidence/accept-out/codex-final4-ffc/independent-main-after.json
[F4-ARCHIVE]: ../../.local/remed-20261113/codex-closeout/final4-archive-receipt.json
[F4-RUNTIME]: ../../.local/remed-20261113/codex-closeout/remote-evidence/accept-out/codex-final4-ffc/runtime-provenance/summary.json
[A2]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-a2-independent-review.json
[B2]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-b2-independent-review.json
[PAIR-INTEGRITY]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-a2-b2-full-integrity.json
[PAIR-EQUALITY]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-a2-b2-cache-byte-equality.json
[PAIR-LEDGER]: ../../.local/remed-20261113/codex-closeout/snapshot-stream-step2/timing-a2-b2-independent-ledger-review.json
[COVERAGE-V5]: ../../.local/remed-20261113/codex-closeout/audit-commit-review-coverage-v5.json
[F4-TIMING]: ../../.local/remed-20261113/codex-closeout/final4-full-all-role-timing-v2.json
[F4-COMPARE]: ../../.local/remed-20261113/codex-closeout/final4-vs-final3-full-comparison.json
