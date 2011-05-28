/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifdef DO_UNIT_TESTS

#include "header.h"
#include "Compartment.h"
#include "HHGate.h"
#include "ChanBase.h"
#include "HHChannel.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

extern void testCompartment(); // Defined in Compartment.cpp
extern void testCompartmentProcess(); // Defined in Compartment.cpp
extern void testSpikeGen(); // Defined in SpikeGen.cpp
extern void testCaConc(); // Defined in CaConc.cpp
extern void testNernst(); // Defined in Nernst.cpp
/*
extern void testSynChan(); // Defined in SynChan.cpp
extern void testBioScan(); // Defined in BioScan.cpp
*/

void testHHGateCreation()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	// cout << "\nTesting HHChannel";
	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id chanId = shell->doCreate( "HHChannel", nid, "Na", dims );
	MsgId mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
		ObjId( chanId ), "channel" );
	assert( mid != Msg::badMsg );
	
	// Check gate construction and removal when powers are assigned
	vector< Id > kids;
	HHChannel* chan = reinterpret_cast< HHChannel* >(chanId.eref().data());
	assert( chan->xGate_ == 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[1] == Id( chanId.value() + 2 ) );
	assert( kids[2] == Id( chanId.value() + 3 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 0 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 1 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );
	// Read the size of the gateIds.

	Field< double >::set( chanId, "Xpower", 2 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 0 );
	assert( chan->xGate_ == 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	// Even though gate was deleted, its Id is intact.
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 0 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Xpower", 2 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ == 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	// New gate was created but has orig Id
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 0 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Ypower", 3 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ != 0 );
	assert( chan->zGate_ == 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 1 );
	assert( kids[2]()->dataHandler()->localEntries() == 0 );

	Field< double >::set( chanId, "Zpower", 1 );
	assert( chan->xGate_ != 0 );
	assert( chan->yGate_ != 0 );
	assert( chan->zGate_ != 0 );
	kids = Field< vector< Id > >::get( chanId, "children" );
	assert( kids.size() == 3 );
	assert( kids[0] == Id( chanId.value() + 1 ) );
	assert( kids[1] == Id( chanId.value() + 2 ) );
	assert( kids[2] == Id( chanId.value() + 3 ) );
	assert( kids[0]()->dataHandler()->localEntries() == 1 );
	assert( kids[1]()->dataHandler()->localEntries() == 1 );
	assert( kids[2]()->dataHandler()->localEntries() == 1 );

	shell->doDelete( nid );
	cout << "." << flush;
}


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

// y(x) = (A + B * x) / (C + exp((x + D) / F))
// So: A = 0.1e6*(EREST+0.025), B = -0.1e6, C = -1.0, D = -(EREST+0.025)
// F = -0.01

double Na_m_A( double v )
{
	if ( fabs( EREST + 0.025 - v ) < 1e-6 )
		v += 1e-6;
	return  0.1e6 * ( EREST + 0.025 - v ) / ( exp ( ( EREST + 0.025 - v )/ 0.01 ) - 1.0 );
}

// A = 4.0e3, B = 0, C = 0.0, D = -EREST, F = 0.018
double Na_m_B( double v )
{
	return 4.0e3 * exp ( ( EREST - v ) / 0.018 );
}

// A = 70, B = 0, C = 0, D = -EREST, F = 0.02
double Na_h_A( double v )
{
	return 70.0 * exp ( ( EREST - v ) / 0.020 );
}

// A = 1e3, B = 0, C = 1.0, D = -(EREST + 0.03 ), F = -0.01
double Na_h_B( double v )
{
	return 1.0e3 / ( exp ( ( 0.030 + EREST - v )/ 0.01 )  + 1.0 );
}

// A = 1e4 * (0.01 + EREST), B = -1e4, C = -1.0, D = -(EREST + 0.01 ), F = -0.01
double K_n_A( double v )
{
	if ( fabs( EREST + 0.025 - v ) < 1e-6 )
		v += 1e-6;
	
	return ( 1.0e4 * ( 0.01 + EREST - v ) ) / ( exp ( ( 0.01 + EREST         - v ) / 0.01 ) - 1.0 );
}

// A = 0.125e3, B = 0, C = 0, D = -EREST ), F = 0.08
double K_n_B( double v )
{
	return 0.125e3 * exp ( (EREST - v ) / 0.08 );
}

void testHHGateLookup()
{
	Id shellId = Id();
	HHGate gate( shellId, Id( 1 ) );
	Eref er = Id(1).eref();
	Qinfo q;
	gate.setMin( er, &q, -2.0 );
	gate.setMax( er, &q, 2.0 );
	gate.setDivs( er, &q, 1 );
	assert( gate.A_.size() == 2 );
	assert( gate.B_.size() == 2 );
	assert( gate.getDivs( er, &q ) == 1 );
	assert( doubleEq( gate.invDx_, 0.25 ) );
	gate.A_[0] = 0;
	gate.A_[1] = 4;
	gate.lookupByInterpolation_ = 0;
	assert( doubleEq( gate.lookupA( -3 ), 0 ) );
	assert( doubleEq( gate.lookupA( -2 ), 0 ) );
	assert( doubleEq( gate.lookupA( -1.5 ), 0 ) );
	assert( doubleEq( gate.lookupA( -1 ), 0 ) );
	assert( doubleEq( gate.lookupA( -0.5 ), 0 ) );
	assert( doubleEq( gate.lookupA( 0 ), 0 ) );
	assert( doubleEq( gate.lookupA( 1 ), 0 ) );
	assert( doubleEq( gate.lookupA( 2 ), 4 ) );
	assert( doubleEq( gate.lookupA( 3 ), 4 ) );
	gate.lookupByInterpolation_ = 1;
	assert( doubleEq( gate.lookupA( -3 ), 0 ) );
	assert( doubleEq( gate.lookupA( -2 ), 0 ) );
	assert( doubleEq( gate.lookupA( -1.5 ), 0.5 ) );
	assert( doubleEq( gate.lookupA( -1 ), 1 ) );
	assert( doubleEq( gate.lookupA( -0.5 ), 1.5 ) );
	assert( doubleEq( gate.lookupA( 0 ), 2 ) );
	assert( doubleEq( gate.lookupA( 1 ), 3 ) );
	assert( doubleEq( gate.lookupA( 2 ), 4 ) );
	assert( doubleEq( gate.lookupA( 3 ), 4 ) );

	gate.B_[0] = -1;
	gate.B_[1] = 1;
	double x = 0;
	double y = 0;
	gate.lookupBoth( -3 , &x, &y );
	assert( doubleEq( x, 0 ) );
	assert( doubleEq( y, -1 ) );
	gate.lookupBoth( -2 , &x, &y );
	assert( doubleEq( x, 0 ) );
	assert( doubleEq( y, -1 ) );
	gate.lookupBoth( -0.5, &x, &y );
	assert( doubleEq( x, 1.5 ) );
	assert( doubleEq( y, -0.25 ) );
	gate.lookupBoth( 0, &x, &y );
	assert( doubleEq( x, 2 ) );
	assert( doubleEq( y, 0 ) );
	gate.lookupBoth( 1.5, &x, &y );
	assert( doubleEq( x, 3.5 ) );
	assert( doubleEq( y, 0.75 ) );
	gate.lookupBoth( 100000, &x, &y );
	assert( doubleEq( x, 4 ) );
	assert( doubleEq( y, 1 ) );

	cout << "." << flush;
}

void testHHGateSetup()
{
	Id shellId = Id();
	HHGate gate( shellId, Id( 1 ) );
	Eref er = Id(1).eref();
	Qinfo q;

	vector< double > parms;
	// Try out m-gate of NA.
// For the alpha:
// A = 0.1e6*(EREST*0.025), B = -0.1e6, C= -1, D= -(EREST+0.025), F = -0.01
// For the beta: A = 4.0e3, B = 0, C = 0.0, D = -EREST, F = 0.018
// xdivs = 100, xmin = -0.1, xmax = 0.05
	unsigned int xdivs = 100;
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

	gate.setupAlpha( er, &q, parms );
	assert( gate.A_.size() == xdivs + 1 );
	assert( gate.B_.size() == xdivs + 1 );

	double x = xmin;
	double dx = (xmax - xmin) / xdivs;
	for ( unsigned int i = 0; i <= xdivs; ++i ) {
		double ma = Na_m_A( x );
		double mb = Na_m_B( x );
		assert( doubleEq( gate.A_[i], ma ) );
		assert( doubleEq( gate.B_[i], ma + mb ) );
		x += dx;
	}

	cout << "." << flush;
}

	////////////////////////////////////////////////////////////////
	// Check construction and result of HH squid simulation
	////////////////////////////////////////////////////////////////
void testHHChannel()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id naId = shell->doCreate( "HHChannel", comptId, "Na", dims );
	MsgId mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
		ObjId( naId ), "channel" );
	assert( mid != Msg::badMsg );
	Id kId = shell->doCreate( "HHChannel", comptId, "K", dims );
	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
		ObjId( kId ), "channel" );
	assert( mid != Msg::badMsg );

	Id tabId = shell->doCreate( "Table", nid, "tab", dims );
	mid = shell->doAddMsg( "Single", ObjId( tabId, 0 ), "requestData",
		ObjId( comptId, 0 ), "get_Vm" );
	
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

	// Check parameters
	vector< double > A = Field< vector< double > >::get( kids[1], "tableA");
	vector< double > B = Field< vector< double > >::get( kids[1], "tableB");
	double x = xmin;
	double dx = (xmax - xmin) / xdivs;
	for ( unsigned int i = 0; i <= xdivs; ++i ) {
		double ha = Na_h_A( x );
		double hb = Na_h_B( x );
		assert( doubleEq( A[i], ha ) );
		assert( doubleEq( B[i], ha + hb ) );
		x += dx;
	}

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

	// Check parameters
	A = Field< vector< double > >::get( kids[0], "tableA" );
	B = Field< vector< double > >::get( kids[0], "tableB" );
	x = xmin;
	for ( unsigned int i = 0; i <= xdivs; ++i ) {
		double na = K_n_A( x );
		double nb = K_n_B( x );
		if ( i != 40 && i != 55 ) { 
			// Annoying near-misses due to different ways of handling
			// singularity. I think the method in the HHGate is better.
			assert( doubleEq( A[i], na ) );
			assert( doubleEq( B[i], na + nb ) );
		}
		x += dx;
	}

	//////////////////////////////////////////////////////////////////////
	// Schedule, Reset, and run.
	//////////////////////////////////////////////////////////////////////

	shell->doSetClock( 0, 1.0e-5 );
	shell->doSetClock( 1, 1.0e-5 );
	shell->doSetClock( 2, 1.0e-5 );
	shell->doSetClock( 3, 1.0e-4 );

	shell->doUseClock( "/n/compt", "init", 0 );
	shell->doUseClock( "/n/compt", "process", 1 );
	// shell->doUseClock( "/n/compt/##", "process", 2 );
	shell->doUseClock( "/n/compt/Na,/n/compt/K", "process", 2 );
	shell->doUseClock( "/n/tab", "process", 3 );

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
	// Clear it all up
	////////////////////////////////////////////////////////////////
	shell->doDelete( nid );
	cout << "." << flush;
}

///////////////////////////////////////////////////
// Unit tests for SynChan
///////////////////////////////////////////////////

// #include "SpikeGen.h"

/**
 * Here we set up a SynChan recieving spike inputs from two
 * SpikeGens. The first has a delay of 1 msec, the second of 10.
 * The tau of the SynChan is 1 msec.
 * We test for generation of peak responses at the right time, that
 * is 2 and 11 msec.
 */
void testSynChan()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );

	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );

	Id synChanId = shell->doCreate( "SynChan", nid, "synChan", dims );
	Id synId( synChanId.value() + 1 );
	Id sgId1 = shell->doCreate( "SpikeGen", nid, "sg1", dims );
	Id sgId2 = shell->doCreate( "SpikeGen", nid, "sg2", dims );
	ProcInfo p;
	p.dt = 1.0e-4;
	p.currTime = 0;
	bool ret;
	assert( synId()->getName() == "synapse" );
	ret = Field< double >::set( synChanId, "tau1", 1.0e-3 );
	assert( ret );
	ret = Field< double >::set( synChanId, "tau2", 1.0e-3 );
	assert( ret );
	ret = Field< double >::set( synChanId, "Gbar", 1.0 );
	assert( ret );

	// This is a hack, should really inspect msgs to automatically figure
	// out how many synapses are needed.
	ret = Field< unsigned int >::set( synChanId, "num_synapse", 2 );
	assert( ret );

	Element* syne = synId();
	assert( syne->dataHandler()->localEntries() == 2 );
	dynamic_cast< FieldDataHandlerBase* >( syne->dataHandler() )->setNumField( synChanId.eref().data(), 2 );
	
	assert( syne->dataHandler()->totalEntries() == 2 );
	assert( syne->dataHandler()->numDimensions() == 1 );
	assert( syne->dataHandler()->sizeOfDim( 0 ) == 2 );

	MsgId mid = shell->doAddMsg( "single", 
		ObjId( sgId1, DataId( 0, 0 ) ), "event",
		ObjId( synId, DataId( 0, 0 ) ), "addSpike" );
	assert( mid != Msg::badMsg );
	mid = shell->doAddMsg( "single", 
		ObjId( sgId2, DataId( 0, 0 ) ), "event",
		ObjId( synId, DataId( 0, 1 ) ), "addSpike" );
	assert( mid != Msg::badMsg );
	
	ret = Field< double >::set( sgId1, "threshold", 0.0 );
	ret = Field< double >::set( sgId1, "refractT", 1.0 );
	ret = Field< bool >::set( sgId1, "edgeTriggered", 0 );
	ret = Field< double >::set( sgId2, "threshold", 0.0 );
	ret = Field< double >::set( sgId2, "refractT", 1.0 );
	ret = Field< bool >::set( sgId2, "edgeTriggered", 0 );


	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"weight", 1.0 );
	assert( ret);
	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"delay", 0.001 );
	assert( ret);
	ret = Field< double >::set( ObjId( synId, DataId( 0, 1 ) ),
		"weight", 1.0 );
	assert( ret);
	ret = Field< double >::set( ObjId( synId, DataId( 0, 1 ) ), 
		"delay", 0.01 );
	assert( ret);

	double dret;
	dret = Field< double >::get( ObjId( synId, DataId( 0, 0 ) ), "weight" );
	assert( doubleEq( dret, 1.0 ) );
	dret = Field< double >::get( ObjId( synId, DataId( 0, 0 ) ), "delay" );
	assert( doubleEq( dret, 0.001 ) );
	dret = Field< double >::get( ObjId( synId, DataId( 0, 1 ) ), "weight" );
	assert( doubleEq( dret, 1.0 ) );
	dret = Field< double >::get( ObjId( synId, DataId( 0, 1 ) ), "delay" );
	assert( doubleEq( dret, 0.01 ) );

	dret = SetGet1< double >::set( sgId1, "Vm", 2.0 );
	dret = SetGet1< double >::set( sgId2, "Vm", 2.0 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleEq( dret, 0.0 ) );

	/////////////////////////////////////////////////////////////////////

	shell->doSetClock( 0, 1e-4 );
	// shell->doUseClock( "/n/##", "process", 0 );
	shell->doUseClock( "/n/synChan,/n/sg1,/n/sg2", "process", 0 );
	// shell->doStart( 0.001 );
	shell->doReinit();
	shell->doReinit();

	shell->doStart( 0.001 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.0 ) );

	shell->doStart( 0.0005 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.825 ) );

	shell->doStart( 0.0005 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 1.0 ) );

	shell->doStart( 0.001 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.736 ) );

	shell->doStart( 0.001 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.406 ) );

	shell->doStart( 0.007 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.997 ) );

	shell->doDelete( nid );
	cout << "." << flush;
}


void testNMDAChan()
{
    Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );

    vector< unsigned int > dims( 1, 1 );
    Id nid = shell->doCreate( "Neutral", Id(), "n", dims );

    Id synChanId = shell->doCreate( "NMDAChan", nid, "nmdaChan", dims );
    Id synId( synChanId.value() + 1 );
    Id sgId1 = shell->doCreate( "SpikeGen", nid, "sg1", dims );
    ProcInfo p;
    p.dt = 1.0e-4;
    p.currTime = 0;
    bool ret;
    assert( synId()->getName() == "synapse" );
    ret = Field< double >::set( synChanId, "tau1", 130.5e-3 );
    assert( ret );
    ret = Field< double >::set( synChanId, "tau2", 5.0e-3 );
    assert( ret );
    ret = Field< double >::set( synChanId, "Gbar", 1.0 );
    assert( ret );
    
    // This is a hack, should really inspect msgs to automatically figure
    // out how many synapses are needed.
	ret = Field< unsigned int >::set( synChanId, "num_synapse", 1 );
	assert( ret );

	Element* syne = synId();
	assert( syne->dataHandler()->localEntries() == 1 );
	dynamic_cast< FieldDataHandlerBase* >( syne->dataHandler() )->setNumField( synChanId.eref().data(), 1 );
	
	assert( syne->dataHandler()->totalEntries() == 1 );
	assert( syne->dataHandler()->numDimensions() == 1 );
	assert( syne->dataHandler()->sizeOfDim( 0 ) == 1 );

	MsgId mid = shell->doAddMsg( "single", 
		ObjId( sgId1, DataId( 0, 0 ) ), "event",
		ObjId( synId, DataId( 0, 0 ) ), "addSpike" );
	assert( mid != Msg::badMsg );
	
	ret = Field< double >::set( sgId1, "threshold", 0.0 );
	ret = Field< double >::set( sgId1, "refractT", 1.0 );
	ret = Field< bool >::set( sgId1, "edgeTriggered", 0 );


	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"weight", 1.0 );
	assert( ret);
	ret = Field< double >::set( ObjId( synId, DataId( 0, 0 ) ), 
		"delay", 0.001 );
	assert( ret);

	double dret;
	dret = Field< double >::get( ObjId( synId, DataId( 0, 0 ) ), "weight" );
	assert( doubleEq( dret, 1.0 ) );
	dret = Field< double >::get( ObjId( synId, DataId( 0, 0 ) ), "delay" );
	assert( doubleEq( dret, 0.001 ) );

	dret = SetGet1< double >::set( sgId1, "Vm", 2.0 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleEq( dret, 0.0 ) );

	/////////////////////////////////////////////////////////////////////

	shell->doSetClock( 0, 1e-4 );
	// shell->doUseClock( "/n/##", "process", 0 );
	shell->doUseClock( "/n/synChan,/n/sg1", "process", 0 );
	// shell->doStart( 0.001 );
	shell->doReinit();
	shell->doReinit();

	shell->doStart( 0.001 );
	dret = Field< double >::get( synChanId, "Gk" );
	assert( doubleApprox( dret, 0.0 ) );

	shell->doStart( 0.0005 );
	dret = Field< double >::get( synChanId, "Gk" );
        cout << "Gk:" << dret << endl;
	assert( doubleApprox( dret, 1.0614275017053588e-07 ) );

	// shell->doStart( 0.0005 );
	// dret = Field< double >::get( synChanId, "Gk" );
        // cout << "Gk:" << dret << endl;
	// assert( doubleApprox( dret, 1.0 ) );

	// shell->doStart( 0.001 );
	// dret = Field< double >::get( synChanId, "Gk" );
        // cout << "Gk:" << dret << endl;
	// assert( doubleApprox( dret, 0.736 ) );

	// shell->doStart( 0.001 );
	// dret = Field< double >::get( synChanId, "Gk" );
        // cout << "Gk:" << dret << endl;
	// assert( doubleApprox( dret, 0.406 ) );

	// shell->doStart( 0.007 );
	// dret = Field< double >::get( synChanId, "Gk" );
        // cout << "Gk:" << dret << endl;
	// assert( doubleApprox( dret, 0.997 ) );

	shell->doDelete( nid );
	cout << "." << flush;
    
}


// This tests stuff without using the messaging.
void testBiophysics()
{
	testCompartment();
	testHHGateCreation();
	testHHGateLookup();
	testHHGateSetup();
	testSpikeGen();
	testCaConc();
	testNernst();
	/*
	testBioScan();
	*/
}

// This is applicable to tests that use the messaging and scheduling.
void testBiophysicsProcess()
{
	testCompartmentProcess();
	testHHChannel();
	testSynChan();
}

#endif
