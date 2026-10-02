#!/usr/bin/env python3
"""bench_cli.py -- the small shared pieces of the eval_<method>.py CLIs.

Every benchmark runner takes `--benchmark {srbench,llmsrbench}` and dispatches to
a run_<bench>() function.  Two things then need saying in each of them, and are
easy to get subtly wrong, so they live here instead:

  bench_defaults()   a flag whose sensible default differs per benchmark parses
                     with default=None, so "unset" stays distinguishable from
                     "given the other benchmark's value"; this fills it in.

  check_bench_flags() a flag only one benchmark reads is an ERROR under the other.
                     Before the per-benchmark scripts were merged, argparse caught
                     this for free (the other script simply had no such flag).
                     slurm/eval_common.sh forwards extra knobs to both benchmarks, so
                     losing that check would silently ignore a knob the caller
                     believed was in effect.

Deliberately import-light (stdlib only): it is imported during argument parsing,
before any model stack is loaded.
"""

import sys


def bench_defaults(args, defaults, keys):
    """Fill in per-benchmark defaults for every key in `keys` left at None.

    `defaults` is {benchmark: {key: value}}.  Returns the benchmark's whole dict
    so the caller can also read entries that are not plain flags.
    """
    d = defaults[args.benchmark]
    for key in keys:
        if getattr(args, key) is None:
            setattr(args, key, d[key])
    return d


def check_bench_flags(args, parser, bench_only, argv=None):
    """Error if a flag belonging to the OTHER benchmark was passed.

    `bench_only` is {benchmark: [dest, ...]}.  Keyed on the option being present
    in argv rather than on its value differing from the default -- `--split
    lsr_transform` is still a mistake under --benchmark srbench even though it
    happens to equal that flag's default.
    """
    argv = sys.argv[1:] if argv is None else argv
    given = {a.split("=", 1)[0] for a in argv if a.startswith("-")}
    opts = {act.dest: act.option_strings for act in parser._actions}
    other = "llmsrbench" if args.benchmark == "srbench" else "srbench"
    offenders = sorted(
        {o for f in bench_only.get(other, []) for o in opts.get(f, []) if o in given})
    if offenders:
        parser.error(f"{', '.join(offenders)}: {other}-only flag(s), but "
                     f"--benchmark is {args.benchmark}.")
