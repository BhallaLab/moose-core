# lm2.py --- 
# Upi Bhalla, NCBS Bangalore 2013.
#
# Commentary: 
# 
# Testing system for loading in arbirary multiscale models based on
# model definition files.
# This is a cleaned up and modular version of loadMulti.py
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 

# Code:

import sys
import os
os.environ['NUMPTHREADS'] = '1'
import signal
import math

import moose
import proto18

EREST_ACT = -70e-3

# These three lines are used to break script for gdb to track.
PID = os.getpid()
def do_nothing( *args ):
    pass

def loadElec( elecFile ):
	library = moose.Neutral( '/library' )
	moose.setCwe( '/library' )
	proto18.make_Ca()
	proto18.make_Ca_conc()
	proto18.make_K_AHP()
	proto18.make_K_C()
	proto18.make_Na()
	proto18.make_K_DR()
	proto18.make_K_A()
	proto18.make_glu()
	proto18.make_NMDA()
	proto18.make_Ca_NMDA()
	proto18.make_NMDA_Ca_conc()
	proto18.make_axon()
	model = moose.Neutral( '/model' )
	cellId = moose.loadModel( elecFile, '/model/elec', "hsolve" )
	return cellId

def addPlot( objpath, field, plot ):
	#assert moose.exists( objpath )
	if moose.exists( objpath ):
		tab = moose.Table( '/graphs/' + plot )
		obj = moose.element( objpath )
		if obj.className == 'Neutral':
			print "addPlot failed: object is a Neutral: ", objpath
			return moose.element( '/' )
		else:
			moose.connect( tab, 'requestData', obj, field )
			return tab
	else:
		print "addPlot failed: object not found: ", objpath
		return moose.element( '/' )

def dumpPlots( fname ):
    if ( os.path.exists( fname ) ):
        os.remove( fname )
    for x in moose.wildcardFind( '/graphs/#[ISA=Table]' ):
        moose.element( x[0] ).xplot( fname, x[0].name )
    for x in moose.wildcardFind( '/graphs/elec/#[ISA=Table]' ):
        moose.element( x[0] ).xplot( fname, x[0].name + '_e' )

def moveCompt( path, oldParent, newParent ):
	meshEntries = moose.element( newParent.path + '/mesh' )
	# Set up vol messaging from new compts to all their child objects.
	for x in moose.wildcardFind( path + '/##[ISA=PoolBase]' ):
		moose.connect( meshEntries, 'mesh', x, 'mesh', 'OneToOne' )
	orig = moose.element( path )
	moose.move( orig, newParent )
	moose.delete( moose.ematrix( oldParent.path ) )
	chem = moose.element( '/model/chem' )
	moose.move( newParent, chem )

def makeChemSolver( compartmentPath ):
	# Put in the solvers, see how they fare.
	solve = moose.GslStoich( compartmentPath + '/ksolve' )
	solve.path = compartmentPath + '/##';
	solve.compartment = moose.element( compartmentPath )
	solve.method = 'rk5'
	nm = moose.element( compartmentPath + '/mesh' )
	moose.connect( nm, 'remesh', solve, 'remesh' )
    #print "neuron: nv=", solve.numLocalVoxels, ", nav=", solve.numAllVoxels, solve.numVarPools, solve.numAllPools
	return solve
	
def loadChem( chemFile, diffLength ):
	neuroCompt = moose.NeuroMesh( '/model/neuroMesh' )
	neuroCompt.separateSpines = 1
	neuroCompt.diffLength = diffLength
	neuroCompt.geometryPolicy = 'cylinder'
	spineCompt = moose.SpineMesh( '/model/spineMesh' )
	moose.connect( neuroCompt, 'spineListOut', spineCompt, 'spineList', 'OneToOne' )
	psdCompt = moose.PsdMesh( '/model/psdMesh' )
	#print 'Meshvolume[neuro, spine, psd] = ', neuroCompt.mesh[0].volume, spineCompt.mesh[0].volume, psdCompt.mesh[0].volume
	moose.connect( neuroCompt, 'psdListOut', psdCompt, 'psdList', 'OneToOne' )

	modelId = moose.loadModel( chemFile, '/model', 'ee' )
	chem = moose.element( '/model/model' )
	chem.name = 'chem'
	oldN = moose.element( '/model/chem/kinetics' )
	oldS = moose.element( '/model/chem/compartment_1' )
	oldP = moose.element( '/model/chem/compartment_2' )
	oldN.volume = neuroCompt.mesh[0].volume
	oldS.volume = spineCompt.mesh[0].volume
	oldP.volume = psdCompt.mesh[0].volume
	moveCompt( '/model/chem/kinetics/PSD', oldP, psdCompt )
	moveCompt( '/model/chem/kinetics/SPINE', oldS, spineCompt )
	moveCompt( '/model/chem/kinetics/DEND', oldN, neuroCompt )

	makeChemSolver( neuroCompt.path )
	makeChemSolver( spineCompt.path )
	makeChemSolver( psdCompt.path )
	return modelId

def loadStimulus( freq, receptorPath, weight, delay ):
	synInput = moose.SpikeGen( '/model/elec/synInput' )
	synInput.threshold = -1.0
	synInput.edgeTriggered = 0
	synInput.Vm( 0 )
	synInput.refractT = 1.0/freq

	for r in moose.wildcardFind( receptorPath ):
		re = moose.element( r.path )
		re.synapse.num = 1 
		syn = moose.element( r.path + '/synapse' )
		moose.connect( synInput, 'event', syn, 'addSpike', 'Single' )
		syn.weight = weight # 0.5
		syn.delay = delay # 1 ms ish.

# generalize this later.
def setAdaptor( chemCaPath):
	psdCompt = moose.element( '/model/chem/psdMesh' )
	pdc = psdCompt.mesh.num
	aCa = moose.Adaptor( '/model/chem/psdMesh/adaptCa', pdc )
	adaptCa = moose.ematrix( '/model/chem/psdMesh/adaptCa' )
	chemCa = moose.ematrix( chemCaPath )
	assert( len( adaptCa ) == pdc )
	assert( len( chemCa ) == pdc )
	for i in range( pdc ):
		path = '/model/elec/spine_head_14_' + str( i + 1 ) + '/NMDA_Ca_conc'
		elecCa = moose.element( path )
		moose.connect( elecCa, 'concOut', adaptCa[i], 'input', 'Single' )
	moose.connect( adaptCa, 'outputSrc', chemCa, 'set_conc', 'OneToOne' )
	adaptCa.inputOffset = 0.0	# 
	adaptCa.outputOffset = 0.00008	# 80 nM offset in chem.
   	adaptCa.scale = 1e-5	# 520 to 0.0052 mM
	"""
	aGluR = moose.Adaptor( '/model/chem/psdMesh/adaptGluR', 5 )
    adaptGluR = moose.ematrix( '/model/chem/psdMesh/adaptGluR' )
	chemR = moose.ematrix( '/model/chem/psdMesh/psdGluR' )
	assert( len( adaptGluR ) == 5 )
	for i in range( 5 ):
    	path = '/model/elec/head' + str( i ) + '/gluR'
		elecR = moose.element( path )
			moose.connect( adaptGluR[i], 'outputSrc', elecR, 'set_Gbar', 'Single' )
    #moose.connect( chemR, 'nOut', adaptGluR, 'input', 'OneToOne' )
	# Ksolve isn't sending nOut. Not good. So have to use requestField.
    moose.connect( adaptGluR, 'requestField', chemR, 'get_n', 'OneToOne' )
    adaptGluR.outputOffset = 1e-7    # pS
    adaptGluR.scale = 1e-6 / 100     # from n to pS

    adaptK = moose.Adaptor( '/model/chem/neuroMesh/adaptK' )
    chemK = moose.element( '/model/chem/neuroMesh/kChan' )
    elecK = moose.element( '/model/elec/compt/K' )
	moose.connect( adaptK, 'requestField', chemK, 'get_conc', 'OneToAll' )
	moose.connect( adaptK, 'outputSrc', elecK, 'set_Gbar', 'Single' )
	adaptK.scale = 0.3               # from mM to Siemens
	"""


def makeNeuroMeshModel( elecFile, chemFile, cellPortion, chemCa ):
	diffLength = 20e-6 # But we only want diffusion over part of the model.
	elec = loadElec( elecFile )
	loadStimulus( 21.0, '/model/elec/spine_head_14_#/glu', 4.0, 0.005 )
	loadChem( chemFile, diffLength )

	###########################################################
	# Set up the rdesigneur interface.
	# A useful breakpoint.
	# os.kill( PID, signal.SIGUSR1)
	neuroCompt = moose.element( '/model/chem/neuroMesh' )
	if ( cellPortion == '' ):
		# This loads the chem pathways into the entire neuronal model
		neuroCompt.cell = elec 
	else:
		# This loads chem only in selected compartments
		neuroCompt.cellPortion( elec, cellPortion )

	nmksolve = moose.element( '/model/chem/neuroMesh/ksolve' )
	smksolve = moose.element( '/model/chem/spineMesh/ksolve' )
	pmksolve = moose.element( '/model/chem/psdMesh/ksolve' )

	# We need to use the spine solver as the master for the purposes of
	# these calculations. This will handle the diffusion calculations
	# between head and dendrite, and between head and PSD.
	smksolve.addJunction( nmksolve )
	smksolve.addJunction( pmksolve )
	#print "spine: nv=", smksolve.numLocalVoxels, ", nav=", smksolve.numAllVoxels, smksolve.numVarPools, smksolve.numAllPools
	#print "psd: nv=", pmksolve.numLocalVoxels, ", nav=", pmksolve.numAllVoxels, pmksolve.numVarPools, pmksolve.numAllPools

	###########################################################
	# set up adaptors
	setAdaptor( chemCa )
	#print adaptCa.outputOffset
	#print adaptCa.scale


def makeElecPlots( compts ):
	elec = moose.Neutral( '/graphs/elec' )
	# Default two plots are for soma.
	addPlot( '/model/elec/soma', 'get_Vm', 'elec/somaVm' )
	addPlot( '/model/elec/soma/Ca_conc', 'get_Ca', 'elec/somaCa' )
	for i in moose.wildcardFind( compts ):
		addPlot( i.path, 'get_Vm', 'elec/' + i[0].name + 'Vm' )
		if moose.exists( i.path + 'Ca_conc' ):
			addPlot( i.path + 'Ca_conc', 'get_Ca', 'elec/' + i[0].name + 'Ca' )
		elif moose.exists( i.path + 'NMDA_Ca_conc' ):
			addPlot( i.path + 'NMDA_Ca_conc', 'get_Ca', 'elec/' + i[0].name + 'Ca' )

def makeChemPlots( compts, index ):
	for i in moose.wildcardFind( compts ):
		for k in moose.wildcardFind( i.path + '/##[ISA=PoolBase]' ):
			name = i[0].parent.name + '_' + k[0].name
			#print i.path, k[index].path, 'get_conc', k[0].name, name
			addPlot( k[index].path, 'get_conc', name )

signal.signal( signal.SIGUSR1, do_nothing)

def testNeuroMeshMultiscale():
	elecDt = 50e-6
	chemDt = 1e-4
	plotDt = 5e-4
	plotName = 'lm2.plot'

	# These are the key parameters that define how to set up the multiscale
	# model. Arguments are: 
	# cellmodel, chemmodel, path_to_fill_with_chem, path_to_use_for_adaptor
	# If the path_to_fill_with_chem is an empty string, then the entire
	# cell model is filled with the chem system.
	makeNeuroMeshModel( 'ca1_asym.p', 'psd_merged31e.g', '/model/elec/lat_14_#,/model/elec/spine_neck#,/model/elec/spine_head#', '/model/chem/psdMesh/PSD/CaM/Ca' )

	# Here we set up plots. These are not really part of the model
	# definition, but useful to have a standard place to define them.
	graphs = moose.Neutral( '/graphs' )
	makeElecPlots( '/model/elec/apical_14,/model/elec/spine_head_14_7' )
	makeChemPlots( '/model/chem/psdMesh/PSD/CaM', 6 )
	makeChemPlots( '/model/chem/psdMesh/PSD/CaMKII_PSD', 6 )
	makeChemPlots( '/model/chem/spineMesh/SPINE/CaM', 6 )
	makeChemPlots( '/model/chem/spineMesh/SPINE/CaMKII_BULK', 6 )

	moose.setClock( 0, elecDt )
	moose.setClock( 1, elecDt )
	moose.setClock( 2, elecDt )
	moose.setClock( 5, chemDt )
	moose.setClock( 6, chemDt )
	moose.setClock( 7, plotDt )
	moose.setClock( 8, plotDt )
	moose.useClock( 0, '/model/elec/##[ISA=Compartment]', 'init' )
	moose.useClock( 1, '/model/elec/##[ISA=SpikeGen]', 'process' )
	moose.useClock( 2, '/model/elec/##[ISA=ChanBase],/model/##[ISA=SynBase],/model/##[ISA=CaConc]','process')
	moose.useClock( 5, '/model/chem/##[ISA=PoolBase],/model/##[ISA=ReacBase],/model/##[ISA=EnzBase]', 'process' )
	moose.useClock( 6, '/model/chem/##[ISA=Adaptor]', 'process' )
	moose.useClock( 7, '/graphs/#', 'process' )
	moose.useClock( 8, '/graphs/elec/#', 'process' )
	moose.useClock( 5, '/model/chem/#Mesh/ksolve', 'init' )
	moose.useClock( 6, '/model/chem/#Mesh/ksolve', 'process' )
	hsolve = moose.HSolve( '/model/elec/hsolve' )
	moose.useClock( 1, '/model/elec/hsolve', 'process' )
	hsolve.dt = elecDt
	hsolve.target = '/model/elec/compt'
	moose.reinit()
	moose.reinit()




	moose.start( 0.5 )
	dumpPlots( plotName )
	print 'All done'


def main():
	testNeuroMeshMultiscale()

if __name__ == '__main__':
	main()

# 
# loadMulti.py ends here.
