#!/usr/bin/env bash
# set -x

python3 -c "import moose;print(moose.__file__)"

function run_str() {
    echo "- Timeit: $*" && python3 -m timeit -s "import moose" "$*"
}

# Loading time.
run_str "import moose"
run_str "a=moose.Neutral('a');moose.delete(a)"
run_str "a=moose.Neutral('a', 10000);moose.delete(a)"
run_str "a1=moose.Neutral('a');a2=moose.element(a1);moose.delete(a1)"
run_str "a1=moose.Neutral('a');a2=moose.element(a1);a1==a2;moose.delete(a1)"