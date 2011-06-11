/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
/*
#include "Compartment.h"
#include "HHGate.h"
#include "ChanBase.h"
#include "SynBase.h"
#include "SynChanBase.h"
#include "Synapse.h"
#include "HHChannel.h"
*/
#include "../shell/Shell.h"
#include "../randnum/randnum.h"


static const double EREST = -0.07;

// AP measured in millivolts wrt EREST, at a sample interval of 
// 100 usec
static double actionPotl[] = { 0,
1.2315, 2.39872, 3.51917, 4.61015, 5.69088, 6.78432, 7.91934, 9.13413,
10.482, 12.0417, 13.9374, 16.3785, 19.7462, 24.7909, 33.0981, 47.967,
73.3833, 98.7377, 105.652, 104.663, 101.815, 97.9996, 93.5261, 88.6248,
83.4831, 78.2458, 73.0175, 67.8684, 62.8405, 57.9549, 53.217, 48.6206,
44.1488, 39.772, 35.4416, 31.0812, 26.5764, 21.7708, 16.4853, 10.6048,
4.30989, -1.60635, -5.965, -8.34834, -9.3682, -9.72711,
-9.81085, -9.78371, -9.71023, -9.61556, -9.50965, -9.39655,
-9.27795, -9.15458, -9.02674, -8.89458, -8.75814, -8.61746,
-8.47254, -8.32341, -8.17008, -8.01258, -7.85093, -7.68517,
-7.51537, -7.34157, -7.16384, -6.98227, -6.79695, -6.60796,
-6.41542, -6.21944, -6.02016, -5.81769, -5.61219, -5.40381,
-5.19269, -4.979, -4.76291, -4.54459, -4.32422, -4.10197,
-3.87804, -3.65259, -3.42582, -3.19791, -2.96904, -2.7394,
-2.50915, -2.27848, -2.04755, -1.81652, -1.58556, -1.3548,
-1.12439, -0.894457, -0.665128, -0.436511, -0.208708, 0.0181928,
};


	////////////////////////////////////////////////////////////////
	// Check construction and result of HH squid simulation
	////////////////////////////////////////////////////////////////
void rtHHnetwork( unsigned int numCopies )
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims, 1 );
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims, 1 );
	Id naId = shell->doCreate( "HHChannel", comptId, "Na", dims, 1 );
	MsgId mid = shell->doAddMsg( "OneToOne", ObjId( comptId ), "channel", 
		ObjId( naId ), "channel" );
	assert( mid != Msg::badMsg );
	Id kId = shell->doCreate( "HHChannel", comptId, "K", dims, 1 );
	mid = shell->doAddMsg( "OneToOne", ObjId( comptId ), "channel", 
		ObjId( kId ), "channel" );
	assert( mid != Msg::badMsg );

	
	//////////////////////////////////////////////////////////////////////
	// set up compartment properties
	//////////////////////////////////////////////////////////////////////

	Field< double >::set( comptId, "Cm", 0.007854e-6 );
	Field< double >::set( comptId, "Ra", 7639.44e3 ); // does it matter?
	Field< double >::set( comptId, "Rm", 424.4e3 );
	Field< double >::set( comptId, "Em", EREST + 0.010613 );
	Field< double >::set( comptId, "inject", 0.1e-6 );
	Field< double >::set( comptId, "initVm", EREST );

	//////////////////////////////////////////////////////////////////////
	// set up Na channel properties
	//////////////////////////////////////////////////////////////////////
	Field< double >::set( naId, "Gbar", 0.94248e-3 );
	Field< double >::set( naId, "Ek", EREST + 0.115 );
	Field< double >::set( naId, "Xpower", 3.0 );
	Field< double >::set( naId, "Ypower", 1.0 );

	//////////////////////////////////////////////////////////////////////
	// set up K channel properties
	//////////////////////////////////////////////////////////////////////
	Field< double >::set( kId, "Gbar", 0.282743e-3 );
	Field< double >::set( kId, "Ek", EREST - 0.012 );
	Field< double >::set( kId, "Xpower", 4.0 );

	//////////////////////////////////////////////////////////////////////
	// set up m-gate of Na.
	//////////////////////////////////////////////////////////////////////
	vector< Id > kids; // These are the HHGates.
	kids = Field< vector< Id > >::get( naId, "children" );
	assert( kids.size() == 3 );
	vector< double > parms;
// For the alpha:
// A = 0.1e6*(EREST*0.025), B = -0.1e6, C= -1, D= -(EREST+0.025), F = -0.01
// For the beta: A = 4.0e3, B = 0, C = 0.0, D = -EREST, F = 0.018
// xdivs = 100, xmin = -0.1, xmax = 0.05
	unsigned int xdivs = 150;
	double xmin = -0.1;
	double xmax = 0.05;
	parms.push_back( 0.1e6 * ( EREST + 0.025 ) );	// A alpha
	parms.push_back( -0.1e6 );				// B alpha
	parms.push_back( -1 );					// C alpha
	parms.push_back( -(EREST + 0.025 ) );	// D alpha
	parms.push_back( -0.01 );				// F alpha

	parms.push_back( 4e3 );		// A beta
	parms.push_back( 0 );		// B beta
	parms.push_back( 0 );		// C beta
	parms.push_back( -EREST );	// D beta
	parms.push_back( 0.018 );	// F beta

	parms.push_back( xdivs );
	parms.push_back( xmin );
	parms.push_back( xmax );

	SetGet1< vector< double > >::set( kids[0], "setupAlpha", parms );
	Field< bool >::set( kids[0], "useInterpolation", 1 );

	//////////////////////////////////////////////////////////////////////
	// set up h-gate of NA.
	//////////////////////////////////////////////////////////////////////
	// Alpha rates: A = 70, B = 0, C = 0, D = -EREST, F = 0.02
	// Beta rates: A = 1e3, B = 0, C = 1.0, D = -(EREST + 0.03 ), F = -0.01
	parms.resize( 0 );
	parms.push_back( 70 );
	parms.push_back( 0 );
	parms.push_back( 0 );
	parms.push_back( -EREST );
	parms.push_back( 0.02 );

	parms.push_back( 1e3 );		// A beta
	parms.push_back( 0 );		// B beta
	parms.push_back( 1 );		// C beta
	parms.push_back( -( EREST + 0.03 ) );	// D beta
	parms.push_back( -0.01 );	// F beta

	parms.push_back( xdivs );
	parms.push_back( xmin );
	parms.push_back( xmax );

	SetGet1< vector< double > >::set( kids[1], "setupAlpha", parms );
	Field< bool >::set( kids[1], "useInterpolation", 1 );

	//////////////////////////////////////////////////////////////////////
	// set up n-gate of K.
	//////////////////////////////////////////////////////////////////////
	// Alpha rates: A = 1e4 * (0.01 + EREST), B = -1e4, C = -1.0, 
	//	D = -(EREST + 0.01 ), F = -0.01
	// Beta rates: 	A = 0.125e3, B = 0, C = 0, D = -EREST ), F = 0.08
	kids = Field< vector< Id > >::get( kId, "children" );
	parms.resize( 0 );
	parms.push_back(  1e4 * (0.01 + EREST) );
	parms.push_back( -1e4 );
	parms.push_back( -1.0 );
	parms.push_back( -( EREST + 0.01 ) );
	parms.push_back( -0.01 );

	parms.push_back( 0.125e3 );		// A beta
	parms.push_back( 0 );		// B beta
	parms.push_back( 0 );		// C beta
	parms.push_back( -EREST );	// D beta
	parms.push_back( 0.08 );	// F beta

	parms.push_back( xdivs );
	parms.push_back( xmin );
	parms.push_back( xmax );

	SetGet1< vector< double > >::set( kids[0], "setupAlpha", parms );
	Field< bool >::set( kids[0], "useInterpolation", 1 );

	//////////////////////////////////////////////////////////////////////
	// Set up SpikeGen and SynChan
	//////////////////////////////////////////////////////////////////////
	Id synChanId = shell->doCreate( "SynChan", comptId, "synChan", dims, 1);
	Id synId( synChanId.value() + 1 );
	Id axonId = shell->doCreate( "SpikeGen", comptId, "axon", dims, 1 );
	bool ret;
	assert( synId()->getName() == "synapse" );
	ret = Field< double >::set( synChanId, "tau1", 1.0e-3 );
	assert( ret );
	ret = Field< double >::set( synChanId, "tau2", 1.0e-3 );
	assert( ret );
	ret = Field< double >::set( synChanId, "Gbar", 0.01 );
	assert( ret );
	ret = Field< double >::set( synChanId, "Ek", 0.0 );
	assert( ret );

	mid = shell->doAddMsg( "OneToOne", 
		ObjId( comptId, DataId( 0, 0 ) ), "VmOut",
		ObjId( axonId, DataId( 0, 0 ) ), "Vm" );
	assert( mid != Msg::badMsg );

	mid = shell->doAddMsg( "OneToOne", 
		ObjId( comptId, DataId( 0, 0 ) ), "channel",
		ObjId( synChanId, DataId( 0, 0 ) ), "channel" );
	assert( mid != Msg::badMsg );

	// This is a hack, should really inspect msgs to automatically figure
	// out how many synapses are needed.
	/*
	ret = Field< unsigned int >::set( synChanId, "num_synapse", 2 );
	assert( ret );
	MsgId mid = shell->doAddMsg( "single", 
		ObjId( axonId, DataId( 0, 0 ) ), "event",
		ObjId( synId, DataId( 0, 0 ) ), "addSpike" );
	assert( mid != Msg::badMsg );
	*/
	
	ret = Field< double >::set( axonId, "threshold", 0.0 );
	ret = Field< double >::set( axonId, "refractT", 0.01 );
	ret = Field< bool >::set( axonId, "edgeTriggered", 1 );

	/*
	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"weight", 1.0 );
	assert( ret);
	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"delay", 0.001 );
	assert( ret);
	*/

	//////////////////////////////////////////////////////////////////////
	// Make a copy with lots of duplicates
	//////////////////////////////////////////////////////////////////////
	Id copyParentId = shell->doCreate( "Neutral", Id(), "copy", dims );
	Id copyId = shell->doCopy( comptId, copyParentId, 
		"comptCopies", numCopies, false, false );
	assert( copyId()->dataHandler()->localEntries() == numCopies );
	assert( copyId()->dataHandler()->numDimensions() == 1 );
	kids = Field< vector< Id > >::get( copyId, "children" );
	assert( kids.size() == 4 );
	assert( kids[0]()->getName() == "Na" );
	assert( kids[1]()->getName() == "K" );
	assert( kids[2]()->getName() == "synChan" );
	assert( kids[3]()->getName() == "axon" );

	assert( kids[0]()->dataHandler()->localEntries() == numCopies );
	assert( kids[1]()->dataHandler()->localEntries() == numCopies );
	assert( kids[2]()->dataHandler()->localEntries() == numCopies );
	assert( kids[3]()->dataHandler()->localEntries() == numCopies );

	////////////////////////////////////////////////////////
	// Check that the HHGate data is accessible in copies.
	////////////////////////////////////////////////////////
	vector< Id > gateKids = Field< vector< Id > >::get( kids[1], "children" );
	assert ( gateKids.size() == 3 );
	assert ( gateKids[0]()->dataHandler() != 0 );
	vector< double > kparms = Field< vector< double > >::get( 
		gateKids[0], "alpha" );
	assert( kparms.size() == 5 );
	for ( unsigned int i = 0; i < 5; ++i ) {
		assert( doubleEq( kparms[i], parms[i] ) );
	}
	////////////////////////////////////////////////////////
	// Check that regular fields are the same in copies.
	////////////////////////////////////////////////////////

	double chanEk = Field< double >::get( 
		ObjId( kids[0], (numCopies * 234)/1000  ), 
		"Ek" ); 
	assert( doubleEq( chanEk, EREST + 0.115 ) );
	chanEk = Field< double >::get( 
		ObjId( kids[1], (numCopies * 567)/1000 ), 
		"Ek" ); 
	assert( doubleEq( chanEk, EREST - 0.012 ) );
	double tau1 = Field< double >::get( 
		ObjId( kids[2], (numCopies * 890)/1000 ), 
		"tau1" ); 
	assert( doubleEq( tau1, 0.001 ) );

	//////////////////////////////////////////////////////////////////////
	// Make table to monitor one of the compartments.
	//////////////////////////////////////////////////////////////////////

	Id tabId = shell->doCreate( "Table", copyParentId, "tab", dims );
	mid = shell->doAddMsg( "single", ObjId( tabId, 0 ), "requestData",
		ObjId( copyId, numCopies/2 ), "get_Vm" );
	assert( mid != Msg::badMsg );

	//////////////////////////////////////////////////////////////////////
	// Schedule, Reset, and run.
	//////////////////////////////////////////////////////////////////////
	shell->doSetClock( 0, 1.0e-5 );
	shell->doSetClock( 1, 1.0e-5 );
	shell->doSetClock( 2, 1.0e-5 );
	shell->doSetClock( 3, 1.0e-4 );

	shell->doUseClock( "/copy/comptCopies", "init", 0 );
	shell->doUseClock( "/copy/comptCopies", "process", 1 );
	shell->doUseClock( "/copy/comptCopies/##", "process", 2 );
	// shell->doUseClock( "/copy/compt/Na,/n/compt/K", "process", 2 );
	shell->doUseClock( "/copy/tab", "process", 3 );

	shell->doReinit();
	shell->doReinit();
	shell->doStart( 0.01 );

	//////////////////////////////////////////////////////////////////////
	// Check output
	//////////////////////////////////////////////////////////////////////
	vector< double > vec = Field< vector< double > >::get( tabId, "vec" );
	assert( vec.size() == 101 );
	double delta = 0;
	for ( unsigned int i = 0; i < 100; ++i ) {
		double ref = EREST + actionPotl[i] * 0.001;
		delta += (vec[i] - ref) * (vec[i] - ref);
	}
	assert( sqrt( delta/100 ) < 0.001 );

	////////////////////////////////////////////////////////////////
	// Connect up the network.
	////////////////////////////////////////////////////////////////
	double connectionProbability = 1.5 / sqrt( numCopies );
	Id synCopyId( kids[2].value() + 1 );

	mid = shell->doAddMsg( "Sparse", kids[3], "event", 
		synCopyId, "addSpike" );
	assert( mid != Msg::badMsg );
	Eref manager = Msg::getMsg( mid )->manager();
	SetGet2< double, long >::set( manager.objId(), "setRandomConnectivity",
		connectionProbability, 1234UL );
	
	// shell->doSyncDataHandler( kids[2], "get_numSynapses", synCopyId );
	shell->doSyncDataHandler( synCopyId );

	// Make it twice as big as expected probability, for safety.
	unsigned int numConnections = 2 * numCopies * sqrt( numCopies );
	mtseed( 1000UL );
	vector< double > weight( numConnections );
	vector< double > delay( numConnections );
	double delayRange = 10e-3;
	double delayMin = 5e-3;
	double weightMax = 0.1;
	for ( unsigned int i = 0; i < numConnections; ++i ) {
		weight[i] = mtrand() * weightMax;
		delay[i] = mtrand() * delayRange + delayMin;
	}
	ret = Field< double >::setVec( synCopyId, "weight", weight );
	assert( ret );
	ret = Field< double >::setVec( synCopyId, "delay", delay );
	assert( ret );

	const double injectRange = 0.1e-6;
	vector< double > inject( numCopies, 0 );
	for ( unsigned int i = 0; i < numCopies; ++i )
		inject[i] = mtrand() * injectRange;
	ret = Field< double >::setVec( copyId, "inject", inject );
	assert( ret );

	//////////////////////////////////////////////////////////////////////
	// Reset, and run again. This time long enough to have lots of 
	// synaptic activity
	//////////////////////////////////////////////////////////////////////
	shell->doReinit();
	shell->doReinit();
	shell->doStart( 0.1 );
	SetGet2< string, string >::set( ObjId( tabId ), 
		"xplot", "hhnet.plot", "hhnet" );
	
	////////////////////////////////////////////////////////////////
	// Clear it all up
	////////////////////////////////////////////////////////////////
	shell->doDelete( copyParentId );
	shell->doDelete( nid );
	cout << "." << flush;
}
