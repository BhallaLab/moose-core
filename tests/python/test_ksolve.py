import sys
import time
import moose

print('Using moose from %s' % moose.__file__ )

compt = moose.CubeMesh( '/compt' )
compt.volume = 1e-21

pools = []
for r in range( 10 ):
    a1 = moose.Pool( '/compt/a1%s' % r )
    a1.concInit = 10
    a2 = moose.Pool( '/compt/a2%s' % r )
    a2.concInit = 5
    b1 = moose.Pool( '/compt/b1%s' % r )
    b1.concInit = 1.0
    b2 = moose.Pool( '/compt/b2%s' % r )
    b2.concInit = 1.0
    r = moose.Reac( '/compt/reac%s'% r )
    moose.connect( r, 'sub', a1, 'reac' )
    moose.connect( r, 'sub', a2, 'reac' )
    moose.connect( r, 'prd', b1, 'reac' )
    moose.connect( r, 'prd', b2, 'reac' )
    r.Kf = 1
    r.Kb = 1
    pools += [ a1, a2, b1, b2 ]

ksolve = moose.Ksolve( '/compt/ksolve' )
stoich = moose.Stoich( '/compt/stoich' )
stoich.compartment = compt
stoich.ksolve = ksolve
stoich.path = '/compt/##' 

moose.reinit()
print pools[0].conc, pools[1].conc
t1 = time.time()
moose.start( 200000, 1 )
print('Time taken %s' % (time.time() - t1 ))
print pools[0].conc, pools[1].conc

