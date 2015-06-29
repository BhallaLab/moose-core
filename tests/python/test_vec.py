import sys
sys.path.append('../../python')
import moose
foo = moose.Pool('/foo1', 500)
print foo
bar = moose.vec('/foo1')
assert len(bar) == 500
