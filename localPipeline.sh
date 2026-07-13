#!/usr/bin/env bash
# Run the repository's local quality gates and always print a final summary.

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
REQUIRED_PYTHON="3.14.6"
PIPELINE_LOG_DIR="${TMPDIR:-/tmp}/sales-slip-scanner-pipeline-$$"
export PYTHONDONTWRITEBYTECODE=1
export COVERAGE_FILE="${PIPELINE_LOG_DIR}/.coverage"

declare -a SUMMARY_LINES=()

VENV_OK=0
PYTHON_OK=0
INSTALL_OK=0
LINT_OK=0
TESTS_OK=0

log() {
    printf '[INFO] %s\n' "$*"
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
}

mark_result() {
    local label="$1"
    local status="$2"
    local details="$3"
    SUMMARY_LINES+=("$(printf '%-16s : %-4s %s' "${label}" "${status}" "${details}")")
}

run_with_log() {
    local log_path="$1"
    shift
    "$@" 2>&1 | tee "${log_path}"
    return "${PIPESTATUS[0]}"
}

extract_ruff_details() {
    local log_path="$1"
    local found_line

    if grep -q "All checks passed" "${log_path}"; then
        printf '%s\n' "0 violations"
        return
    fi
    found_line="$(grep -E "Found [0-9]+ error" "${log_path}" | tail -n 1 || true)"
    printf '%s\n' "${found_line:-see Ruff output}"
}

extract_test_details() {
    local log_path="$1"
    local result_line
    local coverage

    result_line="$(grep -E '[0-9]+ (passed|failed)' "${log_path}" | tail -n 1 || true)"
    coverage="$(awk '$1 == "TOTAL" { print $NF }' "${log_path}" | tail -n 1)"
    if [[ -n "${result_line}" && -n "${coverage}" ]]; then
        printf '%s; %s coverage\n' "${result_line}" "${coverage}"
    elif [[ -n "${result_line}" ]]; then
        printf '%s\n' "${result_line}"
    else
        printf '%s\n' "see pytest output"
    fi
}

print_summary() {
    printf '\n========== Local Pipeline Summary ==========\n'
    local line
    for line in "${SUMMARY_LINES[@]}"; do
        printf '%s\n' "${line}"
    done
    printf '============================================\n'
}

# shellcheck disable=SC2329  # Invoked by the EXIT trap in main.
cleanup() {
    rm -rf "${PIPELINE_LOG_DIR}" "${ROOT_DIR}/build"
    rm -rf "${ROOT_DIR}"/*.egg-info
}

prepare_virtual_environment() {
    if [[ -x "${PYTHON}" ]]; then
        if [[ "$("${PYTHON}" -c 'import platform; print(platform.python_version())')" != \
              "${REQUIRED_PYTHON}" ]]; then
            error "Existing .venv does not use Python ${REQUIRED_PYTHON}. Remove it and rerun."
            return 1
        fi
        log "Using existing virtual environment: ${VENV_DIR}"
        return 0
    fi
    log "Creating virtual environment: ${VENV_DIR}"
    "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"
}

main() {
    local ruff_details=""
    local test_details=""
    local exit_code=1

    mkdir -p "${PIPELINE_LOG_DIR}"
    trap cleanup EXIT

    if [[ "$("${SYSTEM_PYTHON}" -c 'import platform; print(platform.python_version())' 2>/dev/null)" == \
          "${REQUIRED_PYTHON}" ]]; then
        PYTHON_OK=1
        mark_result "Python" "PASS" "${REQUIRED_PYTHON}"
    else
        mark_result "Python" "FAIL" "Python ${REQUIRED_PYTHON} is required"
    fi

    if [[ "${PYTHON_OK}" -eq 1 ]] && prepare_virtual_environment; then
        VENV_OK=1
        mark_result "Virtualenv" "PASS" ".venv is available"
    else
        mark_result "Virtualenv" "FAIL" "Could not create or reuse a Python ${REQUIRED_PYTHON} .venv"
    fi

    if [[ "${VENV_OK}" -eq 1 ]]; then
        log "Installing project and development dependencies from pyproject.toml."
        if run_with_log "${PIPELINE_LOG_DIR}/dependencies.log" \
            "${PYTHON}" -m pip install --quiet "${ROOT_DIR}[dev]"; then
            INSTALL_OK=1
            mark_result "Dependencies" "PASS" "pyproject.toml installed"
        else
            mark_result "Dependencies" "FAIL" "Dependency installation failed"
        fi
    else
        mark_result "Dependencies" "SKIP" "Virtualenv is unavailable"
    fi

    if [[ "${INSTALL_OK}" -eq 1 ]]; then
        log "Running Ruff across all repository Python code."
        if run_with_log "${PIPELINE_LOG_DIR}/ruff.log" \
            "${PYTHON}" -m ruff check --no-cache "${ROOT_DIR}"; then
            LINT_OK=1
            ruff_details="$(extract_ruff_details "${PIPELINE_LOG_DIR}/ruff.log")"
            mark_result "Ruff" "PASS" "${ruff_details}"
        else
            ruff_details="$(extract_ruff_details "${PIPELINE_LOG_DIR}/ruff.log")"
            mark_result "Ruff" "FAIL" "${ruff_details}"
        fi

        log "Running pytest with coverage."
        if run_with_log "${PIPELINE_LOG_DIR}/pytest.log" \
            "${PYTHON}" -m pytest "${ROOT_DIR}/tests" \
            -p no:cacheprovider \
            --tb=short \
            -q \
            --cov=salesSlipScanner \
            --cov=receipt_ocr \
            --cov-report=term-missing \
            --cov-fail-under=80; then
            TESTS_OK=1
            test_details="$(extract_test_details "${PIPELINE_LOG_DIR}/pytest.log")"
            mark_result "Tests+Coverage" "PASS" "${test_details}"
        else
            test_details="$(extract_test_details "${PIPELINE_LOG_DIR}/pytest.log")"
            mark_result "Tests+Coverage" "FAIL" "${test_details}"
        fi
    else
        mark_result "Ruff" "SKIP" "Dependencies are unavailable"
        mark_result "Tests+Coverage" "SKIP" "Dependencies are unavailable"
    fi

    if [[ "${PYTHON_OK}" -eq 1 && "${VENV_OK}" -eq 1 && "${INSTALL_OK}" -eq 1 && \
          "${LINT_OK}" -eq 1 && "${TESTS_OK}" -eq 1 ]]; then
        exit_code=0
        log "localPipeline.sh completed successfully"
    else
        error "localPipeline.sh completed with failing mandatory stage(s)"
    fi

    print_summary
    exit "${exit_code}"
}

main "$@"
