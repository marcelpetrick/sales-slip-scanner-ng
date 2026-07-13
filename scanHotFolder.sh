#!/usr/bin/env bash
# Process new receipt images with the local Ollama model and publish Markdown.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
MODEL="${RECEIPT_MODEL:-qwen3.5:4b}"
HOT_FOLDER="${RECEIPT_HOT_FOLDER:-${ROOT_DIR}/hot_folder}"
REPORT="${RECEIPT_REPORT:-}"
KEEP_ALIVE="${RECEIPT_KEEP_ALIVE:-10m}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    case "${arguments[index]}" in
        --model)
            ((index + 1 < ${#arguments[@]})) || die "--model requires a value"
            MODEL="${arguments[index + 1]}"
            ;;
        --model=*) MODEL="${arguments[index]#*=}" ;;
        --hot-folder)
            ((index + 1 < ${#arguments[@]})) || die "--hot-folder requires a value"
            HOT_FOLDER="${arguments[index + 1]}"
            ;;
        --hot-folder=*) HOT_FOLDER="${arguments[index]#*=}" ;;
        --report)
            ((index + 1 < ${#arguments[@]})) || die "--report requires a value"
            REPORT="${arguments[index + 1]}"
            ;;
        --report=*) REPORT="${arguments[index]#*=}" ;;
        --keep-alive)
            ((index + 1 < ${#arguments[@]})) || die "--keep-alive requires a value"
            KEEP_ALIVE="${arguments[index + 1]}"
            ;;
        --keep-alive=*) KEEP_ALIVE="${arguments[index]#*=}" ;;
    esac
done

REPORT="${REPORT:-${HOT_FOLDER}/receipt-report.md}"

command -v ollama >/dev/null 2>&1 || die "Ollama is not installed or not on PATH."
[[ -x "${PYTHON}" ]] || die "Missing .venv. Run ./localPipeline.sh first."

python_version="$("${PYTHON}" -c 'import platform; print(platform.python_version())')"
[[ "${python_version}" == "3.14.6" ]] || die "Python 3.14.6 is required; .venv has ${python_version}."

ollama list >/dev/null 2>&1 || die "The Ollama server is not reachable. Start it with: ollama serve"
ollama show "${MODEL}" >/dev/null 2>&1 || die "Model '${MODEL}' is missing. Install it with: ollama pull ${MODEL}"

mkdir -p "${HOT_FOLDER}"

printf 'Hot folder : %s\n' "${HOT_FOLDER}"
printf 'Report     : %s\n' "${REPORT}"
printf 'Model      : %s\n' "${MODEL}"

exec "${PYTHON}" "${ROOT_DIR}/salesSlipScanner.py" \
    --model "${MODEL}" \
    --hot-folder "${HOT_FOLDER}" \
    --report "${REPORT}" \
    --keep-alive "${KEEP_ALIVE}" \
    "$@"
