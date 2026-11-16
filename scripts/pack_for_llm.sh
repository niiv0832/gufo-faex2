#!/usr/bin/env bash
# pack_for_llm.sh — компактная упаковка контекста Gufo FAEX2 для LLM-сессий.
# Версия: 2.1-faex2
#
# Читает файлы строго через `git ls-files -z`, жёстко исключает .local/,
# .claude/, секреты, артефакты сборки и чужие модели. По умолчанию пишет
# дамп вне дерева репозитория (${HOME}/gufo_faex2_tmp/).

set -euo pipefail

VERSION="2.1-faex2"

usage() {
  cat <<'EOF'
Использование:
  scripts/pack_for_llm.sh [опции]

Опции:
  --scope <docs|code|faex2>  Режим упаковки (по умолчанию: docs)
                               docs  — ядро документации + усечённый CHANGELOG.md + оглавление docs/faex2/
                               code  — ядро документации + указанные --path
                               faex2 — ядро документации + код и тесты форка (qwen, speculative, serve, functional)
  --path <путь>              Путь (файл или каталог) для --scope code; можно указывать несколько раз
  --out <файл>               Путь к выходному файлу (по умолчанию: ${HOME}/gufo_faex2_tmp/faex2_<scope>_<ts>.md)
  --changelog-lines <N|all>  Лимит строк сверху для docs/faex2/CHANGELOG.md (по умолчанию: 160; 0 — пропустить)
  --context-lines <N|all>    Лимит строк сверху для docs/faex2/CONTEXT.md (по умолчанию: all)
  --full-docs                Включить в --scope docs все файлы docs/faex2/ целиком
  --no-core                  Не добавлять стартовое ядро документации (полезно для --scope code)
  --version                  Показать версию скрипта
  -h, --help                 Показать эту справку

Примеры:
  scripts/pack_for_llm.sh --scope docs
  scripts/pack_for_llm.sh --scope code --path src/models/qwen/hip/mtp --path src/core/speculative
  scripts/pack_for_llm.sh --scope faex2 --changelog-lines 120
EOF
}

SCOPE="docs"
OUT_FILE=""
CHANGELOG_LINES="160"
CONTEXT_LINES="all"
FULL_DOCS=0
INCLUDE_CORE=1
CODE_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      [[ $# -ge 2 ]] || { echo "Ошибка: для --scope требуется значение" >&2; exit 2; }
      SCOPE="$2"
      shift 2
      ;;
    --path)
      [[ $# -ge 2 ]] || { echo "Ошибка: для --path требуется значение" >&2; exit 2; }
      CODE_PATHS+=("$2")
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || { echo "Ошибка: для --out требуется значение" >&2; exit 2; }
      OUT_FILE="$2"
      shift 2
      ;;
    --changelog-lines)
      [[ $# -ge 2 ]] || { echo "Ошибка: для --changelog-lines требуется значение" >&2; exit 2; }
      CHANGELOG_LINES="$2"
      shift 2
      ;;
    --context-lines)
      [[ $# -ge 2 ]] || { echo "Ошибка: для --context-lines требуется значение" >&2; exit 2; }
      CONTEXT_LINES="$2"
      shift 2
      ;;
    --full-docs)
      FULL_DOCS=1
      shift
      ;;
    --no-core)
      INCLUDE_CORE=0
      shift
      ;;
    --version)
      echo "pack_for_llm.sh ${VERSION}"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Неизвестная опция: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SCOPE" in
  docs|code|faex2) ;;
  *)
    echo "Ошибка: недопустимый --scope '$SCOPE' (ожидается docs, code или faex2)" >&2
    exit 2
    ;;
esac

if [[ "$SCOPE" == "code" && ${#CODE_PATHS[@]} -eq 0 ]]; then
  echo "Ошибка: для --scope code укажите хотя бы один --path" >&2
  exit 2
fi

validate_lines_arg() {
  local name="$1"
  local val="$2"
  if [[ "$val" != "all" && ! "$val" =~ ^[0-9]+$ ]]; then
    echo "Ошибка: $name должен быть целым числом >= 0 или 'all' (получено: '$val')" >&2
    exit 2
  fi
}

validate_lines_arg "--changelog-lines" "$CHANGELOG_LINES"
validate_lines_arg "--context-lines" "$CONTEXT_LINES"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if ! REPO_ROOT=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null); then
  echo "Ошибка: не удалось определить корень git-репозитория" >&2
  exit 1
fi

if [[ -z "$OUT_FILE" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  OUT_DIR="${HOME}/gufo_faex2_tmp"
  mkdir -p "$OUT_DIR"
  OUT_FILE="${OUT_DIR}/faex2_${SCOPE}_${TS}.md"
else
  mkdir -p "$(dirname "$OUT_FILE")"
fi

# Проверка безопасности: жёсткое исключение .local/, .claude/, секретов, артефактов и чужих моделей.
is_excluded_path() {
  local p="$1"
  local base
  base=$(basename "$p")
  local lower_base
  lower_base=$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')

  # Служебные/непубличные каталоги
  case "$p" in
    .local/*|*/.local/*|.claude/*|*/.claude/*|.git/*|*/.git/*) return 0 ;;
    build/*|*/build/*|result/*|result-*/*|__pycache__/*|*/__pycache__/*) return 0 ;;
  esac

  # Чужие модели и не относящиеся к Qwen/FAEX2 каталоги моделей
  case "$p" in
    src/models/*)
      case "$p" in
        src/models/qwen/*) ;;
        *) return 0 ;;
      esac
      ;;
    tests/models/*)
      case "$p" in
        tests/models/qwen27b/*) ;;
        *) return 0 ;;
      esac
      ;;
  esac

  # Секреты, ключи, токены, окружение
  case "$lower_base" in
    .env|.env.*|*.pem|*.key|*.p12|*.pfx|*.crt|*.kdbx|id_rsa*|id_ed25519*|.ds_store) return 0 ;;
    *secret*|*token*|*credential*|*password*)
      # Исключаем явные файлы секретов, но не исходники/тесты с упоминанием token в имени
      case "$lower_base" in
        *.cpp|*.hpp|*.h|*.c|*.hip|*.py|*.sh|*.md|*.txt|*.cmake) ;;
        *) return 0 ;;
      esac
      ;;
  esac

  # Артефакты сборки, веса моделей, кэши, архивы, медиа
  case "$lower_base" in
    *.o|*.a|*.so|*.so.*|*.dylib|*.pyc|*.gguf|*.bin|*.safetensors|*.kvc) return 0 ;;
    *.tar|*.tar.*|*.zst|*.gz|*.zip|*.7z|*.png|*.jpg|*.jpeg|*.webp|*.pdf|*.mp4|*.wav) return 0 ;;
  esac

  return 1
}

is_tracked_file() {
  local rel="$1"
  git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1
}

lang_for_file() {
  local p="$1"
  case "$p" in
    *.md) echo "markdown" ;;
    *.sh) echo "bash" ;;
    *.py) echo "python" ;;
    *.cpp|*.hpp|*.h|*.c|*.hip) echo "cpp" ;;
    *.json) echo "json" ;;
    *.nix) echo "nix" ;;
    *.cmake|CMakeLists.txt) echo "cmake" ;;
    *) echo "text" ;;
  esac
}

# Список уже добавленных файлов (через разделитель новой строки для совместимости с bash 3.2)
SEEN_FILES=""

mark_seen() {
  local p="$1"
  SEEN_FILES="${SEEN_FILES}"$'\n'"${p}"$'\n'
}

was_seen() {
  local p="$1"
  case "$SEEN_FILES" in
    *$'\n'"${p}"$'\n'*) return 0 ;;
    *) return 1 ;;
  esac
}

append_file() {
  local rel="$1"
  local max_lines="${2:-all}"

  if was_seen "$rel"; then
    return 0
  fi
  if is_excluded_path "$rel"; then
    return 0
  fi
  if ! is_tracked_file "$rel"; then
    return 0
  fi

  local full="${REPO_ROOT}/${rel}"
  if [[ ! -f "$full" ]]; then
    return 0
  fi

  mark_seen "$rel"

  local total_lines
  total_lines=$(wc -l < "$full" | tr -d ' ')
  local lang
  lang=$(lang_for_file "$rel")

  {
    if [[ "$max_lines" != "all" && "$total_lines" -gt "$max_lines" ]]; then
      echo "## File: ${rel} (показаны первые ${max_lines} из ${total_lines} строк)"
    else
      echo "## File: ${rel}"
    fi
    echo ""
    echo "\`\`\`${lang}"
    if [[ "$max_lines" != "all" && "$total_lines" -gt "$max_lines" ]]; then
      head -n "$max_lines" "$full"
      echo ""
      echo "... [обрезано: показано ${max_lines} из ${total_lines} строк] ..."
    else
      cat "$full"
    fi
    echo ""
    echo "\`\`\`"
    echo ""
  } >> "$OUT_FILE"
}

append_core_docs() {
  local core_files=(
    "AGENTS.md"
    "docs/faex2/CONTEXT.md"
    "docs/faex2/SPEC.md"
    "docs/faex2/DESIGN.md"
    "docs/faex2/ROADMAP_SPEC_01.md"
    "src/models/qwen/AGENTS.md"
  )

  local f
  for f in "${core_files[@]}"; do
    if [[ "$f" == "docs/faex2/CONTEXT.md" ]]; then
      append_file "$f" "$CONTEXT_LINES"
    else
      append_file "$f" "all"
    fi
  done

  if [[ "$CHANGELOG_LINES" != "0" ]]; then
    append_file "docs/faex2/CHANGELOG.md" "$CHANGELOG_LINES"
  fi
}

append_git_ls_prefix() {
  local prefix="$1"
  while IFS= read -r -d '' rel; do
    append_file "$rel" "all"
  done < <(git -C "$REPO_ROOT" ls-files -z -- "$prefix")
}

GIT_BRANCH=$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null || echo "detached")
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
GENERATED_AT=$(date +"%Y-%m-%d %H:%M:%S %z")

cat > "$OUT_FILE" <<EOF
# Gufo FAEX2 Context Pack

- **pack_for_llm.sh:** \`${VERSION}\`
- **Сгенерировано:** ${GENERATED_AT}
- **Ветка / SHA:** \`${GIT_BRANCH}\` / \`${GIT_SHA}\`
- **Режим (--scope):** \`${SCOPE}\`
- **Лимит CHANGELOG.md:** \`${CHANGELOG_LINES}\` строк
- **Лимит CONTEXT.md:** \`${CONTEXT_LINES}\` строк

---

EOF

if [[ "$INCLUDE_CORE" -eq 1 ]]; then
  append_core_docs
fi

case "$SCOPE" in
  docs)
    if [[ "$FULL_DOCS" -eq 1 ]]; then
      append_git_ls_prefix "docs/faex2"
    else
      {
        echo "## Остальные документы в docs/faex2/ (не включены в компактный режим)"
        echo ""
      } >> "$OUT_FILE"
      while IFS= read -r -d '' rel; do
        if ! was_seen "$rel" && ! is_excluded_path "$rel"; then
          sz=$(wc -c < "${REPO_ROOT}/${rel}" | tr -d ' ')
          echo "- \`${rel}\` (${sz} байт)" >> "$OUT_FILE"
        fi
      done < <(git -C "$REPO_ROOT" ls-files -z -- "docs/faex2")
      echo "" >> "$OUT_FILE"
    fi
    ;;
  code)
    for p in "${CODE_PATHS[@]}"; do
      # Убираем возможный префикс ./ для аккуратного сопоставления
      clean_p="${p#./}"
      append_git_ls_prefix "$clean_p"
    done
    ;;
  faex2)
    append_git_ls_prefix "src/models/qwen"
    append_git_ls_prefix "src/core/speculative"
    append_git_ls_prefix "src/cli/serve"
    append_git_ls_prefix "tests/models/qwen27b"
    append_git_ls_prefix "tests/functional"
    append_file "tools/README.md" "all"
    append_file "scripts/pack_for_llm.sh" "all"
    ;;
esac

OUT_BYTES=$(wc -c < "$OUT_FILE" | tr -d ' ')
OUT_LINES=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo "Готово: ${OUT_FILE} (${OUT_BYTES} байт, ${OUT_LINES} строк)"
