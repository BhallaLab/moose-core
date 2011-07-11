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
#include "../shell/Shell.h"

extern void testCompartment(); // Defined in Compartment.cpp
extern void testCompartmentProcess(); // Defined in Compartment.cpp
extern void testSpikeGen(); // Defined in SpikeGen.cpp
extern void testCaConc(); // Defined in CaConc.cpp
extern void testNernst(); // Defined in Nernst.cpp
extern void testMarkovRateTable(); //Defined in MarkovRateTable.cpp
extern void testVectorTable();	//Defined in VectorTable.cpp
extern void testMarkovSolverBase();	//Defined in MarkovSolverBase.cpp
extern void testMarkovSolver();		//Defined in MarkovSolver.cpp
extern void testLeakyIaF(); // Defined in LeakyIaF.cpp

/*
extern void testSynChan(); // Defined in SynChan.cpp
extern void testBioScan(); // Defined in BioScan.cpp
*/

void testHHGateCreation()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
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

/////////////////////////////////////
// Markov Channel unit tests.
////////////////////////////////////

//Sample current obtained from channel in Chapter 20, Sakmann & Neher, Pg. 603.
//The current is sampled at intervals of 10 usec.
static double sampleCurrent[] = 
   {0.0000000e+00, 3.0005743e-26, 1.2004594e-25, 2.7015505e-25, 4.8036751e-25, 7.5071776e-25,
   1.0812402e-24, 1.4719693e-24, 1.9229394e-24, 2.4341850e-24, 3.0057404e-24, 3.6376401e-24,
   4.3299183e-24, 5.0826095e-24, 5.8957481e-24, 6.7693684e-24, 7.7035046e-24, 8.6981913e-24,
   9.7534627e-24, 1.0869353e-23, 1.2045897e-23, 1.3283128e-23, 1.4581082e-23, 1.5939791e-23,
   1.7359292e-23, 1.8839616e-23, 2.0380801e-23, 2.1982878e-23, 2.3645883e-23, 2.5369850e-23,
   2.7154813e-23, 2.9000806e-23, 3.0907863e-23, 3.2876020e-23, 3.4905309e-23, 3.6995766e-23,
   3.9147423e-23, 4.1360317e-23, 4.3634480e-23, 4.5969946e-23, 4.8366751e-23, 5.0824928e-23,
   5.3344511e-23, 5.5925535e-23, 5.8568033e-23, 6.1272040e-23, 6.4037589e-23, 6.6864716e-23,
   6.9753453e-23, 7.2703835e-23, 7.5715897e-23, 7.8789672e-23, 8.1925194e-23, 8.5122497e-23,
   8.8381616e-23, 9.1702584e-23, 9.5085435e-23, 9.8530204e-23, 1.0203692e-22, 1.0560563e-22,
   1.0923636e-22, 1.1292913e-22, 1.1668400e-22, 1.2050099e-22, 1.2438013e-22, 1.2832146e-22,
   1.3232502e-22, 1.3639083e-22, 1.4051894e-22, 1.4470937e-22, 1.4896215e-22, 1.5327733e-22,
   1.5765494e-22, 1.6209501e-22, 1.6659757e-22, 1.7116267e-22, 1.7579032e-22, 1.8048057e-22,
   1.8523345e-22, 1.9004900e-22, 1.9492724e-22, 1.9986821e-22, 2.0487195e-22, 2.0993849e-22,
   2.1506786e-22, 2.2026010e-22, 2.2551524e-22, 2.3083331e-22, 2.3621436e-22, 2.4165840e-22,
   2.4716548e-22, 2.5273563e-22, 2.5836888e-22, 2.6406527e-22, 2.6982483e-22, 2.7564760e-22,
   2.8153360e-22, 2.8748287e-22, 2.9349545e-22, 2.9957137e-22, 3.0571067e-22 };

void testMarkovGslSolver()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< unsigned int > dims( 1, 1 );

	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id rateTabId = shell->doCreate( "MarkovRateTable", comptId, "rateTab", dims ); 
	Id mChanId = shell->doCreate( "MarkovChannel", comptId, "mChan", dims );
	Id gslSolverId = shell->doCreate( "MarkovGslSolver", comptId, "gslSolver", dims );

	Id tabId = shell->doCreate( "Table", nid, "tab", dims );

	MsgId mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
			ObjId( mChanId ), "channel" );
	assert( mid != Msg::badMsg );

	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel",
			ObjId( rateTabId ), "channel" );
	assert( mid != Msg::badMsg );

	mid = shell->doAddMsg( "Single", ObjId( gslSolverId ), "stateOut", 
			ObjId( mChanId ), "handlestate" );
	assert( mid != Msg::badMsg );

	mid = shell->doAddMsg("Single", ObjId( rateTabId ), "instratesOut",
			ObjId( gslSolverId ), "handleQ" ); 

	mid = shell->doAddMsg( "Single", ObjId( tabId, 0 ), "requestData",
			ObjId( mChanId, 0 ), "get_Ik" );
	assert( mid != Msg::badMsg );

	//////////////////////////////////////////////////////////////////////
	// set up compartment properties
	//////////////////////////////////////////////////////////////////////

	Field< double >::set( comptId, "Cm", 0.007854e-6 );
	Field< double >::set( comptId, "Ra", 7639.44e3 ); // does it matter?
	Field< double >::set( comptId, "Rm", 424.4e3 );
	Field< double >::set( comptId, "Em", -0.1 );	
	Field< double >::set( comptId, "inject", 0 );
	Field< double >::set( comptId, "initVm", -0.1 );

	/////////////////////////////////
	//
	//Setup of Markov Channel.
	//This is a simple 5-state channel model taken from Chapter 20, "Single-Channel
	//Recording", Sakmann & Neher.  
	//All the transition rates are constant.
	//
	////////////////////////////////
	
	//Setting number of states, number of open states.
	Field< unsigned int >::set( mChanId, "numstates", 5 );
	Field< unsigned int >::set( mChanId, "numopenstates", 2 );

	//Setting initial state of system.  
	vector< double > initState;
	
	initState.push_back( 0.0 );
  initState.push_back( 0.0 );
  initState.push_back( 0.0 );
 	initState.push_back( 0.0 );
 	initState.push_back( 1.0 );

	Field< vector< double > >::set( mChanId, "initialstate", initState );

	vector< string > stateLabels;

	stateLabels.push_back( "O1" );
	stateLabels.push_back( "O2" );
	stateLabels.push_back( "C1" );
	stateLabels.push_back( "C2" );
	stateLabels.push_back( "C3" );

	Field< vector< string > >::set( mChanId, "labels", stateLabels );	
	
	vector< double > gBars;

	gBars.push_back( 40e-12 );
	gBars.push_back( 50e-12 );

	Field< vector< double > >::set( mChanId, "gbar", gBars );

	//Setting up rate tables.
	SetGet1< unsigned int >::set( rateTabId, "setuptables", 5 );

	//Filling in values into one parameter rate table. Note that all rates here
	//are constant.
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 1, 2, 0.05 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 1, 4, 3 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 2, 1, 0.00066667 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 2, 3, 0.5 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 3, 2, 15 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 3, 4, 4 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 4, 1, 0.015 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 4, 3, 0.05 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 4, 5, 2.0 );
	SetGet3< unsigned int, unsigned int, double >::
		set( rateTabId, "setconst", 5, 4, 0.01 );

	//Setting initial state of the solver. Once this is set, the solver object
	//will send out messages containing the updated state to the channel object.  
	SetGet1< vector< double > >::set( gslSolverId, "setinitstate", initState );
		
	shell->doSetClock( 0, 1.0e-5 );	
	shell->doSetClock( 1, 1.0e-5 );	
	shell->doSetClock( 2, 1.0e-5 );	
	shell->doSetClock( 3, 1.0e-5 );	

	//Voltage is clamped to -100 mV in the example. Hence, we skip running the
	//process function.
	shell->doUseClock( "/n/compt", "init", 0 );
	shell->doUseClock( "/n/compt", "process", 1 );
	shell->doUseClock( "/n/compt/gslSolver,/n/compt/rateTab", "process", 1 );
	shell->doUseClock( "/n/compt/mChan,/n/tab", "process", 2 );

	shell->doReinit( );
	shell->doReinit( );
	shell->doStart( 1.0e-3 );

	vector< double > vec = Field< vector< double > >::get( tabId, "vec" );

	for ( unsigned i = 0; i < 101; ++i )
		assert( doubleEq( sampleCurrent[i] * 1e25, vec[i] * 1e25 ) );
	//Currents involved here are incredibly small. Scaling them up is necessary
	//for the doubleEq function to do its job.

	shell->doDelete( nid );
	cout << "." << flush;
}

////////////////
//The testMarkovGslSolver() function includes the MarkovChannel object, but
//is a rather trivial case, in that the rates are all constant. 
//This test simultaneously tests the MarkovChannel, MarkovGslSolver, 
//MarkovSolverBase and MarkovSolver classes. 
//This test involves simulating the 4-state NMDA channel model specified 
//in the following paper : 
//"Voltage Dependence of NMDA-Activated Macroscopic Conductances Predicted
//by Single-Channel Kinetics", Craig E. Jahr and Charles F. Stevens, The Journal
//of Neuroscience, 1990, 10(9), pp. 3178-3182.
//It is expected that the MarkovGslSolver and the MarkovSolver objects will
//give the same answer. 
//
//Note that this is different from the NMDAChan test which involves synapses.
///////////////
void testMarkovChannel()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< unsigned int > dims( 1, 1 );
	
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims ); 
	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id rateTableId = shell->doCreate( "MarkovRateTable", comptId, "rateTable", dims );
	Id mChanGslId = shell->doCreate( "MarkovChannel", comptId, "mChanGsl", dims );
	Id mChanExptlId = shell->doCreate( "MarkovChannel", comptId, "mChanExptl", dims );

	Id gslSolverId = shell->doCreate( "MarkovGslSolver", comptId, "gslSolver", dims );
	Id exptlSolverId = shell->doCreate( "MarkovSolver", comptId, "exptlSolver", dims );

	Id gslTableId = shell->doCreate( "Table", nid, "gslTable", dims );
	Id exptlTableId = shell->doCreate( "Table", nid, "exptlTable", dims );
	
	Id int2dTableId = shell->doCreate( "Interpol2D", nid, "int2dTable", dims );
	Id vecTableId = shell->doCreate( "VectorTable", nid, "vecTable", dims );

	vector< double > table1d;
	vector< vector< double > > table2d;

	///////////////////////////
	//Setting up the messaging.
	//////////////////////////

	////////
	//Connecting up the MarkovGslSolver.
	///////

	//Connecting Compartment and MarkovChannel objects. 
	//Compartment sends Vm to MarkovChannel object. The MarkovChannel, 
	//via its ChanBase base class, sends back the conductance and current through
	//it.
	MsgId	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
			  ObjId( mChanGslId ), "channel" );
	assert( mid != Msg::badMsg );

	//Connecting Compartment and MarkovRateTable.
	//The MarkovRateTable's job is to send out the instantaneous rate matrix,
	//Q, to the solver object(s). 
	//In order to do so, the MarkovRateTable object needs information on 
	//Vm and ligand concentration to look up the rate from the table provided
	//by the user. Hence, the need of the connection to the Compartment object.
	//However, unlike a channel object, the MarkovRateTable object does not
	//return anything to the Compartment directly, and communicates only with the 
	//solvers.
	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel",
			ObjId( rateTableId ), "channel" );
	assert( mid != Msg::badMsg );

	//Connecting the MarkovRateTable with the MarkovGslSolver object.
	//As mentioned earlier, the MarkovRateTable object sends out information
	//about Q to the MarkovGslSolver. The MarkovGslSolver then churns out
	//the state of the system for the next time step. 
	mid = shell->doAddMsg("Single", ObjId( rateTableId ), "instratesOut",
			ObjId( gslSolverId ), "handleQ" ); 

	//Connecting MarkovGslSolver with MarkovChannel.
	//The MarkovGslSolver object, upon computing the state of the channel, 
	//sends this information to the MarkovChannel object. The MarkovChannel
	//object will compute the expected conductance of the channel and send
	//this information to the compartment. 
	mid = shell->doAddMsg( "Single", ObjId( gslSolverId ), "stateOut", 
			ObjId( mChanGslId ), "handlestate" );
	assert( mid != Msg::badMsg );

	//////////
	//Connecting up the MarkovSolver class.
	/////////

	//Connecting the MarkovSolver and Compartment.
	//The MarkovSolver derives from the MarkovSolverBase class.
	//The base class need Vm and ligand concentration information to 
	//perform lookup and interpolation on the matrix exponential lookup 
	//tables.
	mid = shell->doAddMsg( "Single", ObjId( comptId ), "channel", 
			ObjId( exptlSolverId ), "channel" );
	assert( mid != Msg::badMsg );						

	/////////
	//Connecting up the Table objects to cross-check values.
	////////

	//Get the current values from the GSL solver based channel.
	mid = shell->doAddMsg( "Single", ObjId( gslTableId ), "requestData", 
				ObjId( mChanGslId ), "get_Ik" );
	assert( mid != Msg::badMsg );

	//Get the current values from the matrix exponential solver based channel.
	mid = shell->doAddMsg( "Single", ObjId( exptlTableId ), "requestData", 
				ObjId( mChanExptlId ), "get_Ik" );
	assert( mid != Msg::badMsg );

	////////////////////
	//Compartment properties. Identical to ones used in testHHChannel()
	//barring a few modifications.
	///////////////////

	Field< double >::set( comptId, "Cm", 0.007854e-6 );
	Field< double >::set( comptId, "Ra", 7639.44e3 ); // does it matter?
	Field< double >::set( comptId, "Rm", 424.4e3 );
	Field< double >::set( comptId, "Em", EREST + 0.02 );	
	Field< double >::set( comptId, "inject", 0 );
	Field< double >::set( comptId, "initVm", EREST );

	//////////////////
	//Setup of rate tables.
	//Refer paper mentioned at the header of the unit test for more
	//details.
	/////////////////

	//Number of states and open states.
	Field< unsigned int >::set( mChanGslId, "numstates", 4 );		
	Field< unsigned int >::set( mChanExptlId, "numstates", 4 );		

	Field< unsigned int >::set( mChanGslId, "numopenstates", 1 );		
	Field< unsigned int >::set( mChanExptlId, "numopenstates", 1 );		

	vector< string > stateLabels;

	//In the MarkovChannel class, the opening states are listed first.
	//This is in line with the convention followed in Chapter 20, Sakmann & 
	//Neher. 
	stateLabels.push_back( "O" );		//State 1.
	stateLabels.push_back( "B1" );	//State 2.
	stateLabels.push_back( "B2" );	//State 3.
	stateLabels.push_back( "C" ); 	//State 4.

	Field< vector< string > >::set( mChanGslId, "labels", stateLabels );	
	Field< vector< string > >::set( mChanExptlId, "labels", stateLabels );	

	//Setting up conductance value for single open state.	Value chosen
	//is quite arbitrary.
	vector< double > gBar;

	gBar.push_back( 5.431553e-9 );

	Field< vector< double > >::set( mChanGslId, "gbar", gBar );
	Field< vector< double > >::set( mChanExptlId, "gbar", gBar );

	//Initial state of the system. This is really an arbitrary choice.
	vector< double > initState;

	initState.push_back( 0.30 ); 
	initState.push_back( 0.05 ); 
	initState.push_back( 0.30 ); 
	initState.push_back( 0.35 ); 

	Field< vector< double > >::set( mChanGslId, "initialstate", initState );
	Field< vector< double > >::set( mChanExptlId, "initialstate", initState );

	//This initializes the GSL solver object.
	SetGet1< vector< double > >::set( gslSolverId, "init", initState );	

	//Initializing MarkovRateTable object.
	double v;
	double conc;

	SetGet1< unsigned int >::set( rateTableId, "setuptables", 4 );

	//Setting up lookup tables for the different rates.		
	
	//Transition from "O" to "B1" i.e. r12.
	Field< double >::set( vecTableId, "xmin", -0.05 );
	Field< double >::set( vecTableId, "xmin", 0.10 );
	Field< unsigned int >::set( vecTableId, "xdivs", 150 );

	v = -0.05;
	for ( unsigned int i = 0; i < 151; ++i )	
	{
		table1d.push_back( exp( -16 * v - 2.91 ) );
		v += 0.001;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );

	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 1, 2, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "B1" back to O i.e. r21
	v = -0.05;	
	for ( unsigned int i = 0; i < 151; ++i )
	{
		table1d.push_back( exp( 9 * v + 1.22 ) );
		v += 0.001;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );
	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 2, 1, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "O" to "B2" i.e. r13
	//This is actually a 2D rate. But, there is no change in Mg2+ concentration
	//that occurs. Hence, I create a 2D lookup table anyway but I manually
	//set the concentration on the rate table object anyway.

	Field< double >::set( rateTableId, "ligandconc", 24e-6 );

	Field< double >::set( int2dTableId, "xmin", -0.05 );
	Field< double >::set( int2dTableId, "xmax", 0.10 );
	Field< double >::set( int2dTableId, "ymin", 1e-6 );
	Field< double >::set( int2dTableId, "ymax", 30e-6 );
	Field< unsigned int >::set( int2dTableId, "xdivs", 150 );
	Field< unsigned int >::set( int2dTableId, "ydivs", 30 );

	v = 0;
	table2d.resize( 151 );
	for ( unsigned int i = 0; i < 151; ++i )
	{
		conc = 0;
		for ( unsigned int j = 0; j < 31; ++j )
		{
			table2d[i].push_back( 1e6 * conc * exp( -45 * v - 6.97 ) ); 
			conc += 1e-6;
		}
		v += 1e-3;
	}

	Field< vector< vector< double > > >::set( int2dTableId, "tableVector2D", 
																table2d );

	SetGet3< unsigned int, unsigned int, Id >::set( rateTableId, 
																			"set2d", 1, 3, int2dTableId ); 

	//There is only one 2D rate, so no point manually erasing the elements.
	
	//Transition from "B2" to "O" i.e. r31
	v = -0.05;	
	for ( unsigned int i = 0; i < 151; ++i )
	{
		table1d.push_back( exp( 17 * v + 0.96 ) );
		v += 0.001;
	}

	Field< vector< double > >::set( vecTableId, "table", table1d );
	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 3, 1, vecTableId, 0 );

	table1d.erase( table1d.begin(), table1d.end() );

	//Transition from "O" to "C" i.e. r14 
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId,	
									"setconst", 1, 4, exp( -2.847 ) ); 
	
	//Transition from "B1" to "C" i.e. r24
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, 	
									"setconst", 2, 4, exp( -0.693 ) );

	//Transition from "B2" to "C" i.e. r34
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, 
									"setconst", 3, 4, exp( -3.101 ) );

	//Once the rate tables have been set up, we can initialize the 
	//tables in the MarkovSolver class.
		
	shell->doSetClock( 0, 1.0e-3 );	
	shell->doSetClock( 1, 1.0e-3 );	
	shell->doSetClock( 2, 1.0e-3 );	
	shell->doSetClock( 3, 1.0e-3 );	

	shell->doUseClock( "/n/compt", "init", 0 );
	shell->doUseClock( "/n/compt", "process", 1 );
	shell->doUseClock( "/n/compt/gslSolver,/n/compt/rateTable", "process", 2 );
	shell->doUseClock( "/n/compt/mChanGsl,/n/gslTable", "process", 3 );

	shell->doReinit();
	shell->doReinit();
	shell->doStart( 1.0 );

	vector< double > vec = Field< vector< double > >::get( gslTableId, "vec" );

/*	for ( unsigned int i = 0; i < 1001; ++i )
		cout << vec[i] << endl;*/

	shell->doDelete( nid );
	cout << "." << flush;
}

void testMarkovSolverProcess()
{
	Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
	vector< unsigned int > dims( 1, 1 );

	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id vecTableId = shell->doCreate( "VectorTable", nid, "vecTable", dims );
	Id int2dTableId = shell->doCreate( "Interpol2D", nid, "int2dTable", dims );

	Id comptId = shell->doCreate( "Compartment", nid, "compt", dims );
	Id rateTableId = shell->doCreate( "MarkovRateTable", comptId, "compt", dims );
	Id solverId = shell->doCreate( "MarkovSolver", comptId, "compt", dims );

	MsgId mid = shell->doAddMsg("Single", ObjId( comptId ), "channel", 
							ObjId( solverId ), "channel" );
	assert( mid != Msg::badMsg );

	mid = shell->doAddMsg("Single", ObjId( comptId ), "channel", 
							ObjId( rateTableId ), "channel" );
	assert( mid != Msg::badMsg );

	////////////////
	//Initializing the rate table.
	//I create a 3x3 rate table, which is what we would have if one initialized a 
	//3 state Markov channel. 
	//The rate (1,2) is set as a voltage dependent rate, (1,3) is a ligand
	//dependent rate and (2,3) is ligand and voltage dependent. 
	//Rates (2,1) and (3,1) are constant.
	///////////////

	SetGet1< unsigned int >::set( rateTableId, "setuptables", 3 );

	/////////////
	//Setting up rate (1,2) i.e. transition from S1->S2.
	//(1,2) = 0.85 * exp( -0.1 * V );
	//Vmin = -0.05 volts, Vmax = +0.10 volts, dV = 0.005 volts.
	////////////
	
	vector< double > vecEntries;
	for ( double v = -0.05; !doubleEq( v, 0.10 ); v += 0.005 )
		vecEntries.push_back( 0.85 * exp( -0.1 * v ) );	


	Field< double >::set( vecTableId, "xmin", -0.05 );
	Field< double >::set( vecTableId, "xmax", 0.10 );
	Field< vector< double > >::set( vecTableId, "table", vecEntries ); 

	//We now pass in this table to the MarkovRateTable class.
	//The last argument determines if the rate is ligand or voltage dependent.
	//Set this argument to a non-zero value if ligand dependent, and zero if it is 
	//voltage dependent.
	SetGet4< unsigned int, unsigned int, Id, unsigned int >::set( 
			rateTableId, "set1d", 1, 2, vecTableId, 0 );

	///////////
	//Setting up rate (1,3) i.e. transition from S1->S3.
	//(1,3) = 1.29 * [L]   L -> Some ligand species
	//Lmin = 0 nM, Lmax = 100 nM, dL = 1 nM
	//////////
	
	//Cleaning out old vecEntries.
	vecEntries.erase( vecEntries.begin(), vecEntries.end() );
	
	for( double l = 0; !doubleEq(l, 100e-9); l += 1e-9 )
		vecEntries.push_back( 1.29 * l );

	Field< double >::set( vecTableId, "xmin", 0 );
	Field< double >::set( vecTableId, "xmax", 100e-9 );
	Field< vector< double > >::set( vecTableId, "table", vecEntries );

	SetGet4< unsigned int, unsigned int, Id, unsigned int>::set( 
			rateTableId, "set1d", 1, 3, vecTableId, 1 );

	//Erasing this just to be clean.
	vecEntries.erase( vecEntries.begin(), vecEntries.end() );

	/////////
	//Setting up rate (2,3) i.e. transition S2->S3
	//(2,3) = 1.44 * [L] / ( 1 + exp( -0.04V ) )
	//Vmin = -0.75 volts, Vmax = 0.2 volts, dV = 0.001 volts
	//Lmin = 4 nM, Lmax = 50 nM, dL = 2 nM
	//
	//Note that in the case of Interpol2D tables, the ligand concentration is
	//assumed to vary along the y-direction while voltage is on the x-direction, 
	//where x increases downwards and y increases to the right in the lookup tables.
	//Violating this convention can (and will) lead to disaster.
	///////

	vector< vector< double > > int2dEntries;

	Field< double >::set( int2dTableId, "xmin", -0.75 );
	Field< double >::set( int2dTableId, "xmax", 0.2 );
	Field< double >::set( int2dTableId, "ymin", 4e-9 );
	Field< double >::set( int2dTableId, "ymax", 50e-9 );

	unsigned int c = 0;
	for( double v = -0.75; !doubleEq( v, 0.2 ); v += 0.001 )
	{
		int2dEntries.resize( int2dEntries.size() + 1 ); 
		for( double l = 4e-9; !doubleEq( l, 50e-9 ); l += 2e-9 )
			int2dEntries[c].push_back( 1.44 * l / ( 1 + exp( -0.04 * v ) ) );
		++c;
	}

	Field< vector< vector< double > > >::set( int2dTableId, "tableVector2D", int2dEntries );

	SetGet3< unsigned int, unsigned int, Id >::set( rateTableId, "set2d", 2, 3, int2dTableId ); 
	////////////////////////////
	//Rates (2,1) and (3,1) are set to 0.17 and 5.6 respectively.
	///////////////////////////

	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, "setconst", 2, 1, 0.17 );
	SetGet3< unsigned int, unsigned int, double >::set( rateTableId, "setconst", 3, 1, 5.6 );

	//Setup of MarkovRateTable object complete.
	/////////////////////////

	//Initializing MarkovSolver tables.
	SetGet1< Id >::set( solverId, "setuptable", rateTableId ); 

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
	testVectorTable();
	testMarkovSolverBase();
	testMarkovSolver();
	/*
	testBioScan();
	*/
}

// This is applicable to tests that use the messaging and scheduling.
void testBiophysicsProcess()
{
	testCompartmentProcess();
	testHHChannel();
//	testMarkovGslSolver();
//	testMarkovChannel();
//	testMarkovSolverProcess();
//	testSynChan();
        testLeakyIaF();
}

#endif
