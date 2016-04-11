import sys
import moose

print('Using moose from %s' % moose.__file__ )

compt = moose.CubeMesh( '/compt' )
compt.volume = 1e-21
a = moose.Pool( '/compt/a' )
a.concInit = 1.0
b = moose.Pool( '/compt/b' )
b.concInit = 1.0

r = moose.Reac( '/compt/reac' )

moose.connect( r, 'sub', a, 'reac' )
moose.connect( r, 'prd', b, 'reac' )
r.Kf = 1
r.Kb = 10

ksolve = moose.Ksolve( '/compt/ksolve' )
stoich = moose.Stoich( '/compt/stoich' )
stoich.compartment = compt
stoich.ksolve = ksolve
stoich.path = '/compt/##' 

moose.reinit()
moose.start( 1 )

