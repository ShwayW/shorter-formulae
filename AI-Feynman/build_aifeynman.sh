#!/bin/bash
# build_aifeynman.sh -- compile AI Feynman's Fortran brute-force engines.
#
# WHY THIS EXISTS (do not "just pip install ." instead)
# -----------------------------------------------------
# The upstream setup.py builds the seven .f90 files as f2py extension modules via
# `numpy.distutils`, which was REMOVED in NumPy 1.26 (this repo's env/ has NumPy 2.x
# on Python 3.12), so `pip install .` dies at import time with
#     ModuleNotFoundError: No module named 'numpy.distutils'
#
# We don't actually need the f2py wrappers.  Each .f90 is already a standalone
# Fortran *program*:
#     program symbolic_regress
#     call go
#     end
# and aifeynman/S_brute_force.py invokes them as EXECUTABLES on $PATH --
# `subprocess.call(["feynman_sr_mdl_mult"], timeout=...)` -- passing arguments
# through ./args.dat in the cwd, not through Python.  So compiling each program to
# a plain binary named after its console-script entry point is a faithful build,
# and it sidesteps f2py/meson/numpy.distutils entirely.
#
# The entry-point -> source mapping below is copied verbatim from setup.py's
# `console_scripts`; keep them in sync if upstream ever changes.
#
# USAGE
#   ./AI-Feynman/build_aifeynman.sh              # installs into env/bin (venv PATH)
#   OUT=/some/dir ./AI-Feynman/build_aifeynman.sh
#
# On the Alliance/cluster clusters, `module load gcc` provides gfortran.
# On this workstation it is NOT installed by default:  sudo apt install gfortran
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/aifeynman"
OUT="${OUT:-$(cd "$HERE/.." && pwd)/env/bin}"

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found on PATH." >&2
    echo "  Ubuntu/Debian : sudo apt install gfortran" >&2
    echo "  Lmod HPC      : module load gcc" >&2
    exit 1
fi

mkdir -p "$OUT"

# entry-point name : source file   (from setup.py console_scripts)
PROGRAMS=(
    "feynman_sr1:symbolic_regress1.f90"
    "feynman_sr2:symbolic_regress2.f90"
    "feynman_sr3:symbolic_regress3.f90"
    "feynman_sr_mdl_mult:symbolic_regress_mdl3.f90"
    "feynman_sr_mdl_plus:symbolic_regress_mdl2.f90"
    "feynman_sr_mdl4:symbolic_regress_mdl4.f90"
    "feynman_sr_mdl5:symbolic_regress_mdl5.f90"
)

# -ffree-line-length-none : two lines in symbolic_regress_mdl3.f90 exceed the
#                           132-column free-form default and would be truncated.
# -fno-range-check        : the sources build large integer/bit constants that trip
#                           the default compile-time range check.
# -std=legacy             : tolerates the tabs and the GNU `system()`/`lnblnk()`
#                           extensions these files rely on.
# Compiled from inside $SRC so the `include "tools.f90"` lines resolve.
FFLAGS=(-O3 -ffree-line-length-none -fno-range-check -std=legacy -w)

cd "$SRC"
for entry in "${PROGRAMS[@]}"; do
    name="${entry%%:*}"
    src="${entry##*:}"
    echo "  gfortran ${src}  ->  ${OUT}/${name}"
    gfortran "${FFLAGS[@]}" -o "$OUT/$name" "$src"
done

echo
echo "Built ${#PROGRAMS[@]} executables into $OUT"
echo "Verify with:  eval_aifeynman.py --benchmark llmsrbench --check-install"
