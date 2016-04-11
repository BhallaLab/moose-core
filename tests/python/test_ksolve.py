import sys
import moose

print('Using moose from %s' % moose.__file__ )

compt = moose.CubeMesh( '/compt' )
compt.volume = 1e-21
a = moose.Pool( '/compt/a' )
a.concInit = 1.0
b = moose.Pool( '/compt/b' )
b.concInit = 0.5

r = moose.Reac( '/compt/reac' )

moose.connect( r, 'sub', a, 'reac' )
moose.connect( r, 'prd', b, 'reac' )
r.Kf = 10
r.Kb = 1

ksolve = moose.Ksolve( '/compt/ksolve' )
stoich = moose.Stoich( '/compt/stoich' )
stoich.compartment = compt
stoich.ksolve = ksolve
stoich.path = '/compt/##' 

moose.reinit()
print a.conc, b.conc
moose.start( 2 )
print a.conc, b.conc

