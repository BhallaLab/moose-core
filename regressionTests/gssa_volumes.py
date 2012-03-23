# Loads a kkit model, runs it with stimuli, and dumps output.
from moose import *

def dumpPlots( filename, vol ):
	for x in wildcardFind( '/model/graphs/#/#,/model/moregraphs/#/#' ):
		x.xplot( filename, x.name + '.' + str( vol ) )


def loadAndRun( method, volseq ):
	modelId = loadModel( 'AabXJacb.g', 'model', method )

	setClock( 0, 1.0 )
	setClock( 1, 1.0 )
	setClock( 2, 1.0 )
	setClock( 3, 1.0 )

	compt = element( '/model/kinetics' )
	a = element( '/model/kinetics/a' )
	reac = element( '/model/kinetics/kreac' )
	enz = element( '/model/kinetics/c/kenz' )
	le( '/model/kinetics' )

	for vol in volseq:
		compt.buildDefaultMesh( vol, 1 )
		print vol, compt.size, a.concInit, a.nInit
		print vol, 'Reac rate in conc units: (', reac.Kf, reac.Kb, '), n units: (', reac.kf, reac.kb, ')'
		print vol, 'Enz rate in conc units: (', enz.Km, enz.kcat, enz.ratio, '), n units: (', enz.k1, enz.k2, enz.k3, ')'
		reinit()
		start( 100 )
		dumpPlots( 'gssa_volumes' + method + '.plot', vol )
		print 'done', vol
	delete( modelId )

loadAndRun( 'gsl', [1e-18] )
loadAndRun( 'gssa', [1e-21, 1e-20, 1e-19, 1e-18, 1e-17] )

# quit()
