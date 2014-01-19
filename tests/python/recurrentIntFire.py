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
	runtime = 1.0
	delayMin = timestep
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
	network.vec.bufferTime = [delayMax * 2] * size
	moose.le( '/network' )
	network.vec.numSynapses = [1] * size
	# Interesting. This fails because we haven't yet allocated
	# the synapses. I guess it is fair to avoid instances of objects that
	# don't have allocations.
	#synapse = moose.element( '/network/synapse' )
	sv = moose.vec( '/network/synapse' )
	print 'before connect t = ', time.time() - t0
	mid = moose.connect( network, 'spikeOut', sv, 'addSpike', 'Sparse')
	print 'after connect t = ', time.time() - t0
	#print mid.destFields
	m2 = moose.element( mid )
	m2.setRandomConnectivity( connectionProbability, 5489 )
	print 'after setting connectivity, t = ', time.time() - t0
	network.vec.Vm = [(Vmax*random.random()) for r in range(size)]
	network.vec.thresh = thresh
	network.vec.refractoryPeriod = refractoryPeriod
	numSynVec = network.vec.numSynapses
	print 'Middle of setup, t = ', time.time() - t0
	numTotSym = sum( numSynVec )
	for item in network.vec:
		neuron = moose.element( item )
		neuron.synapse.delay = [ (delayMin + random.random() * delayMax) for r in range( len( neuron.synapse ) ) ] 
		neuron.synapse.weight = nprand.rand( len( neuron.synapse ) ) * weightMax
	print 'after setup, t = ', time.time() - t0

	"""

	netvec = network.vec
	for i in range( size ):
		synvec = netvec[i].synapse.vec
		synvec.weight = [ (random.random() * weightMax) for r in range( synvec.len )] 
		synvec.delay = [ (delayMin + random.random() * delayMax) for r in range( synvec.len )] 
	"""

	moose.useClock( 0, '/network', 'process' )
	moose.setClock( 0, timestep )
	moose.setClock( 9, timestep )
	t1 = time.time()
	moose.reinit()
	print 'reinit time t = ', time.time() - t1
	network.vec.Vm = [(Vmax*random.random()) for r in range(size)]
	print 'setting Vm , t = ', time.time() - t1
	t1 = time.time()
	print 'starting'
	moose.start(runtime)
	print 'runtime, t = ', time.time() - t1
	print network.vec.Vm[100:103], network.vec.Vm[900:903]

make_network()
