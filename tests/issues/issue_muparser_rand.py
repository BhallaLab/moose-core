import sys
sys.path.append( '../../_build/moose' )
import moose
print( 'Using moose from %s' % moose.__file__ )

f = moose.Function( '/f' )
f.expr = 'rand(1)'
moose.reinit()
for i in range( 10 ):
    moose.start( 1 )
    print f.value
