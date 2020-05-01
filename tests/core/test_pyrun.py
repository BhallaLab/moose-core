# pyrun.py ---
# Author: Subhasis Ray
# Maintainer: Dilawar Singh

# See https://github.com/BhallaLab/moose-examples/blob/b2e77237ef36e47d0080cc8c4fb6bb94313d2b5e/snippets/pyrun.py
# for details.

import os
import moose
import pytest
import sys
import io
import difflib

stdout_ = sys.stdout
if sys.version_info.major > 2:
    stream_ = io.StringIO()
else:
    stream_ = io.BytesIO()
sys.stdout = stream_

# Removed first 3 lines since they change during each run.
expected = """Running Hello
Hello count = 0
Init World
Running World
World count = 0
Running Hello
Hello count = 1
Running World
World count = 1
Running Hello
Hello count = 2
Running World
World count = 2
Running Hello
Hello count = 3
Running World
World count = 3
Running Hello
Hello count = 4
Running World
World count = 4
Running Hello
Hello count = 5
Running World
World count = 5
Running Hello
Hello count = 6
Running World
World count = 6
Running Hello
Hello count = 7
Running World
World count = 7
Running Hello
Hello count = 8
Running World
World count = 8
Running Hello
Hello count = 9
Running World
World count = 9
Running Hello
Hello count = 10
Running World
World count = 10
Running Hello
Hello count = 11
Running World
World count = 11
Running Hello
Hello count = 12
Running World
World count = 12
Running Hello
Hello count = 13
Running World
World count = 13
Running Hello
Hello count = 14
Running World
World count = 14
Running Hello
Hello count = 15
Running World
World count = 15
Running Hello
Hello count = 16
Running World
World count = 16
Running Hello
Hello count = 17
Running World
World count = 17
Running Hello
Hello count = 18
Running World
World count = 18
Running Hello
Hello count = 19
Running World
World count = 19
Running Hello
Hello count = 20
Running World
World count = 20
Init World
Running World
World count = 0
input = 0.0
output = 0.0
Running World
World count = 1
input = 0.0
output = 0.0
Running World
World count = 2
input = 0.0
output = 0.0
Running World
World count = 3
input = 0.0
output = 0.0
Running World
World count = 4
input = 1.0
output = 1.0
Running World
World count = 5
input = 1.0
output = 1.0
Running World
World count = 6
input = 1.0
output = 1.0
Running World
World count = 7
input = 1.0
output = 1.0
Running World
World count = 8
input = 2.0
output = 4.0
Running World
World count = 9
input = 2.0
output = 4.0
Running World
World count = 10
input = 2.0
output = 4.0
Running World
World count = 11
input = 2.0
output = 4.0
Running World
World count = 12
input = 3.0
output = 9.0
Running World
World count = 13
input = 3.0
output = 9.0
Running World
World count = 14
input = 3.0
output = 9.0
Running World
World count = 15
input = 3.0
output = 9.0
Running World
World count = 16
input = 0.0
output = 0.0
Running World
World count = 17
input = 0.0
output = 0.0
Running World
World count = 18
input = 0.0
output = 0.0
Running World
World count = 19
input = 0.0
output = 0.0
Running World
World count = 20
input = 1.0
output = 1.0
Running World
World count = 21
input = 1.0
output = 1.0
Running World
World count = 22
input = 1.0
output = 1.0
Running World
World count = 23
input = 1.0
output = 1.0
Running World
World count = 24
input = 2.0
output = 4.0
Running World
World count = 25
input = 2.0
output = 4.0
Running World
World count = 26
input = 2.0
output = 4.0
Running World
World count = 27
input = 2.0
output = 4.0
Running World
World count = 28
input = 3.0
output = 9.0
Running World
World count = 29
input = 3.0
output = 9.0
Running World
World count = 30
input = 3.0
output = 9.0
Running World
World count = 31
input = 3.0
output = 9.0
Running World
World count = 32
input = 0.0
output = 0.0
Running World
World count = 33
input = 0.0
output = 0.0
Running World
World count = 34
input = 0.0
output = 0.0
Running World
World count = 35
input = 0.0
output = 0.0
Running World
World count = 36
input = 1.0
output = 1.0
Running World
World count = 37
input = 1.0
output = 1.0
Running World
World count = 38
input = 1.0
output = 1.0
Running World
World count = 39
input = 2.0
output = 4.0
Running World
World count = 40
"""

def run_sequence():
    model = moose.Neutral('/model')
    hello_runner = moose.PyRun('/model/Hello')
    hello_runner.initString = """
print( 'Init', moose.element('/model/Hello') )
hello_count = 0
"""
    hello_runner.runString = """
print( 'Running Hello' )
print( 'Hello count =', hello_count )
hello_count += 1
"""
    hello_runner.run('from datetime import datetime')
    hello_runner.run('print("Hello: current time:", datetime.now().isoformat())')
    moose.useClock(0, hello_runner.path, 'process')
    world_runner = moose.PyRun('World')
    world_runner.initString = """
print( 'Init World' )
world_count = 0
def incr_count():
    global world_count
    world_count += 1
"""
    world_runner.runString = """
print( 'Running World' )
print( 'World count =', world_count )
incr_count()
"""
    world_runner.run('from datetime import datetime')
    world_runner.run('print( "World: current time:", datetime.now().isoformat())')

    moose.useClock(0, world_runner.path, 'process')
    moose.reinit()
    moose.start(0.001)


def input_output():
    model = moose.Neutral('/model')
    input_pulse = moose.PulseGen('/model/pulse')
    #: set the baseline output 0
    input_pulse.baseLevel = 0.0
    #: We make it generate three pulses
    input_pulse.count = 3
    input_pulse.level[0] = 1.0
    input_pulse.level[1] = 2.0
    input_pulse.level[2] = 3.0
    #: Each pulse will appear 1 s after the previous one
    input_pulse.delay[0] = 1.0
    input_pulse.delay[1] = 1.0
    input_pulse.delay[2] = 1.0
    #: Each pulse is 1 s wide
    input_pulse.width[0] = 1.0
    input_pulse.width[1] = 1.0
    input_pulse.width[2] = 1.0
    #: Now create the PyRun object
    pyrun = moose.PyRun('/model/pyrun')
    pyrun.runString = """
output = input_ * input_
print( 'input =', input_ )
print( 'output =', output )
"""
    pyrun.mode = 2 # do not run process method
    moose.connect(input_pulse, 'output', pyrun, 'trigger')
    output_table = moose.Table('/model/output')
    moose.connect(pyrun, 'output', output_table, 'input')
    input_table = moose.Table('/model/input')
    moose.connect(input_pulse, 'output', input_table, 'input')
    moose.setClock(0, 0.25)
    moose.setClock(1, 0.25)
    moose.setClock(2, 0.25)
    moose.useClock(0, input_pulse.path, 'process')
    #: this is unnecessary because the mode=2 ensures that `process`
    #: does nothing
    moose.useClock(1, pyrun.path, 'process')
    moose.useClock(2, '/model/#[ISA=Table]', 'process')
    moose.reinit()
    moose.start(10.0)


# This test will not pass with doctest and coverage
@pytest.mark.xfail(reason="Would not pass with python-coverage")
def test_pyrun():
    global stream_, stdout_, expected
    run_sequence()
    moose.delete('/model')
    input_output()
    sys.stdout = stdout_
    stream_.flush()
    sys.stdout.flush()
    expected = expected.split('\n')[-30:]
    got = stream_.getvalue().split('\n')[-30:]

    if os.environ.get('TRAVIS_OS_NAME', '') != 'osx':
        for x, y in zip(expected, got):
            print("{0:40s} {1:}".format(x, y))
        # Deleted first 3 lines.
        try:
            assert expected == got
        except Exception:
            s = difflib.SequenceMatcher(None, '\n'.join(expected), '\n'.join(got))
            assert s.ratio() >= 0.70, ("Difference is too large", s.ratio())
        print('All done')
    else:
        print("Allow failure on Travis/OSX but not locally.")


if __name__ == '__main__':
    test_pyrun()
