#!/usr/bin/env bash
# localPipeline.sh — run all quality gates before committing
#
# Stages:
#   1. Dependency check
#   2. Lint  (ruff)
#   3. Unit tests with coverage
#   4. Coverage threshold enforcement (80 %)
#
# Exit code: 0 = all green, non-zero = at least one stage failed.
# Run from the repository root: ./localPipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "${GREEN}✔ $*${RESET}"; }
fail() { echo -e "${RED}✖ $*${RESET}"; }
info() { echo -e "${YELLOW}▶ $*${RESET}"; }
sep()  { echo -e "${BOLD}────────────────────────────────────────${RESET}"; }

FAILURES=0

run_stage() {
    local label="$1"; shift
    sep
    info "Stage: ${label}"
    if "$@"; then
        pass "${label} passed"
    else
        fail "${label} FAILED"
        FAILURES=$(( FAILURES + 1 ))
    fi
}

# ── 1. Dependencies ──────────────────────────────────────────────────────────
run_stage "Dependency install" \
    pip install --quiet --break-system-packages -r requirements.txt

# ── 2. Lint ──────────────────────────────────────────────────────────────────
run_stage "Lint (ruff)" \
    ruff check salesSlipScanner.py tests/

# ── 3. Unit tests + coverage ─────────────────────────────────────────────────
sep
info "Stage: Unit tests + coverage"
if python -m pytest tests/ \
        --tb=short \
        -q \
        --cov=salesSlipScanner \
        --cov-report=term-missing \
        --cov-report=html:coverage_html \
        --cov-fail-under=80; then
    pass "Tests + coverage passed"
else
    fail "Tests + coverage FAILED"
    FAILURES=$(( FAILURES + 1 ))
fi

# ── Summary ──────────────────────────────────────────────────────────────────
sep
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All pipeline stages passed.${RESET}"
    echo -e "  Coverage report: ${REPO_ROOT}/coverage_html/index.html"
    exit 0
else
    echo -e "${RED}${BOLD}${FAILURES} stage(s) failed — fix before committing.${RESET}"
    exit 1
fi
