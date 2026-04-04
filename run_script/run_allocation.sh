#!/usr/bin/env bash
# Run from repo root:
#   bash run_script/run_allocation.sh
#       # standard smoke (Time): N=10, prop vs greedy — time + objective in logs and .mat (see README)
#   bash run_script/run_allocation.sh --extended
#       # larger preset: Time at N=20, compares prop, ga, greedy
#   bash run_script/run_allocation.sh --eval-case Performance
#   bash run_script/run_allocation.sh --methods prop,ga --eval-case Time
#
# Methods (see README):
#   prop      — full MIP solved with Gurobi (heavy; N ≤ 20 in this harness)
#   ga        — genetic algorithm (PyGAD; N ≤ 20)
#   greedy    — proposed fast greedy (solve_greedy); not N-capped like prop/ga
#   no_save, all_save — simple fixed policies
#
# Gurobi MIP and GA wall-clock grows very quickly with N; greedy stays fast at large N.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

if ! python -c "from source_alloc.subproblem_solver import my_subproblem_solver_cy" 2>/dev/null; then
  echo "Building Cython extensions in source_alloc/ ..."
  (cd "${ROOT}/source_alloc" && python setup_cython.py build_ext --inplace)
fi

exec python process/allocation.py "$@"
