import moose

compt = moose.CubeMesh( '/compt' )
compt.volume = 1e-20

a = moose.Pool( '/compt/a')
a.concInit = 1.0
b = moose.Pool( '/compt/b' )
b.concInit = 0.6
r = moose.Reac( '/compt/reac' )
r.Kf = 1.0
r.Kb = 0.2

moose.connect( r, 'sub', a, 'reac' )
moose.connect( r, 'prd', b, 'reac' )

stoich = moose.Stoich( '/compt/stoich' )
ksolve = moose.Ksolve( '/compt/ksolve' )
stoich.ksolve = ksolve 
stoich.compartment = compt
stoich.path = '/compt/##'

moose.reinit()
print a.conc, b.conc
moose.start( 1.0 )
print a.conc, b.conc

