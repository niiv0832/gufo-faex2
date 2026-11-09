# AGENTS.md

## Правила собственного форка FAEX2 — читать первыми

Этот репозиторий — `niiv0832/gufo-faex2`, узкая конфигурация Qwen3.6-35B-A3B + спекуляция + зрение на FAEX2 128 ГБ. Следующие правила уточняют унаследованные ниже инструкции для нашей ветки; ограничения безопасности не ослаблять.

1. В начале каждой сессии: `git status`, затем [CONTEXT](docs/faex2/CONTEXT.md), последние записи [журнала форка](docs/faex2/CHANGELOG.md), [SPEC](docs/faex2/SPEC.md), [DESIGN](docs/faex2/DESIGN.md). При работе с владельческим стендом прочитать `.local/HANDOFF.md`, если он есть; отсутствие локального файла не повод угадывать доступы.
2. Общение и собственная документация — на русском, с объяснениями для начинающего. Имена API/флагов не переводить. Сообщения коммитов: Conventional Commit с русским описанием, например `docs(faex2): оформлена точка продолжения`.
3. Код/тесты/дизайн форка ведутся здесь. Истина о реальной машине, SSH/firewall/BIOS/ОС и deploy — в отдельном эксплуатационном проекте, указанном в локальной памятке. **Перед любой работой с машиной прочитать его AGENTS и текущий контекст.** Если доступа к ним нет, остановиться; сеть/sshd/firewall/ядро/питание не менять по догадке. `sudo` и секреты выполняет/вводит только владелец.
4. Новый агент не должен автоматически запускать сборки, модели, обновления или GPU/NPU-тесты. Изменяющие/длительные действия сначала объяснить, длительные выполнять устойчиво к разрыву SSH. Предыдущий рабочий бинарник сохранять.
5. В рамках поручения коммитить каждый завершённый логический шаг, только свои пути. После изменения обновить CONTEXT и журнал форка. При изменении машины также обновить её эксплуатационные документы. Исторические планы — не команды на повторное исполнение.
6. Без отдельного явного согласия: не делать push, PR/публикацию, rebase/amend, reset --hard, clean, откат чужих незакоммиченных файлов, force-push или удаление веток/данных. Однократная очистка личного email завершена; не запускать её повторно и не merge старую серверную линию обратно.
7. Git-идентичность не менять самовольно. Использовать уже настроенный локальный noreply; перед коммитом/публикацией проверить автора/committer. Не stage `.local/`, бэкапы, старые патчи с личными метаданными, семейные изображения/ответы, ключи или токены. Не использовать `git add -f` для обхода защиты.
8. Для нашего проекта Git — достаточный рабочий инструмент; установка `jj`, `gh`, Nix не является автоматическим требованием. При отсутствии `gh` допустимы согласованные действия владельца в браузере и стандартный Git. Общие upstream-инструкции сборки ниже не заменяют manifest конкретного стенда. Упоминания main/PR в методике ниже означают явно выбранные контрольную ревизию и кандидата; main не является автоматическим baseline нашей ветки.
9. Корневой CHANGELOG и release workflow принадлежат исходной линии. Наш журнал — `docs/faex2/CHANGELOG.md`; автоматический выпуск/Actions не включать без отдельного проектирования. NPU остаётся экспериментом; неизвестное качество не выдавать за lossless.

## Platform and build

Production targets Linux x86-64 AMD Strix Halo (`gfx1151`) only. CMake owns
compiler flags, dependencies and installation; Nix supplies the pinned toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

Text API functional/regression tests live in [tests/functional/](tests/functional/README.md).
Select suites explicitly. Model-specific changes need the affected model/modes;
use all four profiles only for shared behavior changes. Run correctness on the
candidate; use main for matched affected timing controls and suspected regressions,
not a duplicate full correctness run. Reuse baseline evidence only when revision,
toolchain, harness, model, settings and cache history match. Include `long-context`
and `cache` for continuation changes; verify loaded mode and actual drafts.
Correctness and expected prefill/cache work are strict; timings are
mandatory per request/phase at 5% and 3 ms, never averaged across requests.
Run once, investigate flags, then alternate main/PR only for affected histories;
use unchanged-main controls when needed and retain every result. Fix confirmed
regressions before publishing. Noisy evidence stays visibly inconclusive and
unqualified; do not widen margins or stop at reporting failures. Use the code
diff to select affected metrics; unrelated timing variance does not justify
another full matrix.
For reported coding-agent loops, also replay the affected workflow with
`tests/functional/pi_agent.py`; retain its sessions and inspect actual tool results.
Retain numerical quality tests and the standard speed benchmark. Keep these
tests in `tests/functional/`, outside hosted model CI; avoid full sweeps.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use the shared Qwen control-token constants for framing; see
  [src/models/qwen/AGENTS.md](src/models/qwen/AGENTS.md). Inspect their definitions
  when reviewing or changing behavior, and keep reference fixtures independent.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message. Title pull requests
  according to [the release policy](docs/RELEASING.md); squash merging retains
  that title on `main`. Mark breaking changes with `!` in the title, and leave
  version and changelog updates to the release workflow.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
