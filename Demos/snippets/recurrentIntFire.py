#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
# This snippet sets up a recurrent network of IntFire objects, using
# SimpleSynHandlers to deal with spiking events. 
# It isn't very satisfactory as activity runs down after a while.
# It is a good example for using the IntFire, setting up random
# connectivity, and using SynHandlers.
#
import os
import random
import time
from numpy import random as nprand
import sys
sys.path.append('/home/subha/src/moose_async13/python')
import moose

def make_network():
	size = 1024
	timestep = 0.2
	runsteps = 5
	delayMin = 0
	delayMax = 4
	weightMax = 0.02
	Vmax = 1.0
	thresh = 0.8
	refractoryPeriod = 0.4
	connectionProbability = 0.1
	random.seed( 123 )
	nprand.seed( 456 )
	t0 = time.time()

	network = moose.IntFire( 'network', size );
	syns = moose.SimpleSynHandler( '/network/syns', size );
        moose.connect( syns, 'activationOut', network, 'activation', 'OneToOne' )
	moose.le( '/network' )
	syns.vec.numSynapses = [1] * size
	sv = moose.vec( '/network/syns/synapse' )
	print 'before connect t = ', time.time() - t0
	mid = moose.connect( network, 'spikeOut', sv, 'addSpike', 'Sparse')
	print 'after connect t = ', time.time() - t0
	#print mid.destFields
	m2 = moose.element( mid )
	m2.setRandomConnectivity( connectionProbability, 5489 )
	print 'after setting connectivity, t = ', time.time() - t0
	#network.vec.Vm = [(Vmax*random.random()) for r in range(size)]
	network.vec.Vm = nprand.rand( size ) * Vmax
	network.vec.thresh = thresh
	network.vec.refractoryPeriod = refractoryPeriod
	numSynVec = syns.vec.numSynapses
	print 'Middle of setup, t = ', time.time() - t0
	numTotSyn = sum( numSynVec )
        print numSynVec.size, ', tot = ', numTotSyn,  ', numSynVec = ', numSynVec
	for item in syns.vec:
		sh = moose.element( item )
                sh.synapse.delay = delayMin +  (delayMax - delayMin ) * nprand.rand( len( sh.synapse ) )
		#sh.synapse.delay = [ (delayMin + random.random() * (delayMax - delayMin ) for r in range( len( sh.synapse ) ) ] 
		sh.synapse.weight = nprand.rand( len( sh.synapse ) ) * weightMax
	print 'after setup, t = ', time.time() - t0

	"""

	netvec = network.vec
	for i in range( size ):
		synvec = netvec[i].synapse.vec
		synvec.weight = [ (random.random() * weightMax) for r in range( synvec.len )] 
		synvec.delay = [ (delayMin + random.random() * delayMax) for r in range( synvec.len )] 
	"""

	#moose.useClock( 0, '/network/syns,/network', 'process' )
	moose.useClock( 0, '/network/syns', 'process' )
	moose.useClock( 1, '/network', 'process' )
	moose.setClock( 0, timestep )
	moose.setClock( 1, timestep )
	moose.setClock( 9, timestep )
	t1 = time.time()
	moose.reinit()
	print 'reinit time t = ', time.time() - t1
	network.vec.Vm = nprand.rand( size ) * Vmax
	print 'setting Vm , t = ', time.time() - t1
	t1 = time.time()
	print 'starting'
	moose.start(runsteps * timestep)
	print 'runtime, t = ', time.time() - t1
	print network.vec.Vm[99:103], network.vec.Vm[900:903]

make_network()
