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
#include "HHChannel.h"
#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

extern void testCompartment(); // Defined in Compartment.cpp
extern void testCompartmentProcess(); // Defined in Compartment.cpp
/*
extern void testHHChannel(); // Defined in HHChannel.cpp
extern void testCaConc(); // Defined in CaConc.cpp
extern void testNernst(); // Defined in Nernst.cpp
extern void testSpikeGen(); // Defined in SpikeGen.cpp
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

// A = 0.125e3, B = 0, C = 0, D = -EREST ), F = -0.08
double K_n_B( double v )
{
	return 0.125e3 * exp ( (EREST - v ) / 0.08 );
}

void testHHGateLookup()
{
	Id shellId = Id();
	HHGate gate( shellId );
	Eref er = shellId.eref();
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
	HHGate gate( shellId );
	Eref er = shellId.eref();
	Qinfo q;

	vector< double > parms;
	// Try out m-gate of NA.
// For the alpha:
// A = 0.1e6*(EREST*0.025), B = -0.1e6, C= -1, D= -(EREST+0.025), F = -0.01
// beta: A = 4.0e3, B = 0, C = 0.0, D = -EREST, F = 0.018
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

void testHHChannel()
{
	/*
	////////////////////////////////////////////////////////////////
	// Do the Reinit.
	////////////////////////////////////////////////////////////////
	set< double >( chan, "Gbar", 1.0 );
	set< double >( chan, "Ek", 0.0 );
	ProcInfo pb;
	pb.dt = 0.001;
	SetConn c( chan, 0 );
	HHChannel* Na = static_cast< HHChannel* >( c.data() );
	Na->Vm_ = 0.0;

	// This function should do all the reinit steps.
	HHChannel::reinitFunc( &c, &pb );
	ASSERT( Na->Gk_ == 80, "Gk_" );
	ASSERT( Na->X_ == 2, "X_" );
	ASSERT( Na->Y_ == 10, "Y_" );

	////////////////////////////////////////////////////////////////
	// Check construction and result of HH squid simulation
	////////////////////////////////////////////////////////////////
	
	Element* kchan = Neutral::create( "HHChannel", "K", compt->id(), 
		Id::scratchId() );

	ret = Eref( compt ).add( "channel", kchan, "channel" );
	ASSERT( ret, "Setting up K channel" );

	// ASSERT( compt->findFinfo( "channel" )->add( compt, kchan, kchan->findFinfo( "channel" ) ), "Setting up K channel" );

	static const double VMIN = -0.1;
	static const double VMAX = 0.05;
	static const unsigned int XDIVS = 150;

	set< double >( compt, "Cm", 0.007854e-6 );
	set< double >( compt, "Ra", 7639.44e3 ); // does it matter?
	set< double >( compt, "Rm", 424.4e3 );
	set< double >( compt, "Em", EREST + 0.010613 );
	set< double >( compt, "inject", 0.1e-6 );
	set< double >( chan, "Gbar", 0.94248e-3 );
	set< double >( chan, "Ek", EREST + 0.115 );
	set< double >( kchan, "Gbar", 0.282743e-3 );
	set< double >( kchan, "Ek", EREST - 0.012 );
	set< double >( kchan, "Xpower", 4.0 );

	Id kGateId;
	ret = lookupGet< Id, string >( kchan, "lookupChild", kGateId, "xGate" );
	ASSERT( ret, "Look up kGate");
	ASSERT( !kGateId.zero() && !kGateId.bad(), "Lookup kGate" );

	Element* kGate = kGateId();
	ret = lookupGet< Id, string >( kGate, "lookupChild", temp, "A" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "kGate_A" );
	Element* kGate_A = temp();
	ret = lookupGet< Id, string >( kGate, "lookupChild", temp, "B" );
	ASSERT( ret, "Check gate table" );
	ASSERT( !temp.zero() && !temp.bad(), "kGate_B" );
	Element* kGate_B = temp();

	ret = set< double >( xGate_A, "xmin", VMIN ) ; assert( ret );
	ret = set< double >( xGate_B, "xmin", VMIN ) ; assert( ret );
	ret = set< double >( yGate_A, "xmin", VMIN ) ; assert( ret );
	ret = set< double >( yGate_B, "xmin", VMIN ) ; assert( ret );
	ret = set< double >( kGate_A, "xmin", VMIN ) ; assert( ret );
	ret = set< double >( kGate_B, "xmin", VMIN ) ; assert( ret );

	ret = set< double >( xGate_A, "xmax", VMAX ) ; assert( ret );
	ret = set< double >( xGate_B, "xmax", VMAX ) ; assert( ret );
	ret = set< double >( yGate_A, "xmax", VMAX ) ; assert( ret );
	ret = set< double >( yGate_B, "xmax", VMAX ) ; assert( ret );
	ret = set< double >( kGate_A, "xmax", VMAX ) ; assert( ret );
	ret = set< double >( kGate_B, "xmax", VMAX ) ; assert( ret );

	ret = set< int >( xGate_A, "xdivs", XDIVS ) ; assert( ret );
	ret = set< int >( xGate_B, "xdivs", XDIVS ) ; assert( ret );
	ret = set< int >( yGate_A, "xdivs", XDIVS ) ; assert( ret );
	ret = set< int >( yGate_B, "xdivs", XDIVS ) ; assert( ret );
	ret = set< int >( kGate_A, "xdivs", XDIVS ) ; assert( ret );
	ret = set< int >( kGate_B, "xdivs", XDIVS ) ; assert( ret );

	ret = set< int >( xGate_A, "mode", 1 ) ; assert( ret );
	ret = set< int >( xGate_B, "mode", 1 ) ; assert( ret );
	ret = set< int >( yGate_A, "mode", 1 ) ; assert( ret );
	ret = set< int >( yGate_B, "mode", 1 ) ; assert( ret );
	ret = set< int >( kGate_A, "mode", 1 ) ; assert( ret );
	ret = set< int >( kGate_B, "mode", 1 ) ; assert( ret );

	double v = VMIN;
	double dv = ( VMAX - VMIN ) / XDIVS;
	const Finfo* table = xGate_A->findFinfo( "table" );
	for (unsigned int i = 0 ; i <= XDIVS; i++ ) {
		lset( xGate_A, table, Na_m_A( v ), i );
		lset( xGate_B, table, Na_m_A( v ) + Na_m_B( v ), i );
		lset( yGate_A, table, Na_h_A( v ), i );
		lset( yGate_B, table, Na_h_A( v ) + Na_h_B( v ), i );
		lset( kGate_A, table, K_n_A( v ), i );
		lset( kGate_B, table, K_n_A( v ) + K_n_B( v ), i );
		v = v + dv;
	}

	ret = set< double >( compt, "initVm", EREST ); assert( ret );

	pb.dt = 1.0e-5;
	pb.currTime_ = 0.0;
	SetConn c1( compt, 0 );
	SetConn c2( chan, 0 );
	SetConn c3( kchan, 0 );

	moose::Compartment::reinitFunc( &c1, &pb );
	HHChannel::reinitFunc( &c2, &pb );
	HHChannel::reinitFunc( &c3, &pb );

	unsigned int sample = 0;
	double delta = 0.0;
	for ( pb.currTime_ = 0.0; pb.currTime_ < 0.01;
			pb.currTime_ += pb.dt )
	{
		moose::Compartment::processFunc( &c1, &pb );
		HHChannel::processFunc( &c2, &pb );
		HHChannel::processFunc( &c3, &pb );
		if ( static_cast< int >( pb.currTime_ * 1e5 ) % 10 == 0 ) {
			get< double >( compt, "Vm", v );
			// cout << v << endl;
			v -= EREST + actionPotl[ sample++ ] * 0.001;
			delta += v * v;
		}
	}

	ASSERT( delta < 5e-4, "Action potl unit test\n" );
	*/
	
	////////////////////////////////////////////////////////////////
	// Clear it all up
	////////////////////////////////////////////////////////////////
	// shell->doDelete( nid );
	cout << "." << flush;
}

// This tests stuff without using the messaging.
void testBiophysics()
{
	testCompartment();
	testHHGateCreation();
	testHHGateLookup();
	testHHGateSetup();
	/*
	testCaConc();
	testNernst();
	testSpikeGen();
	testSynChan();
	testBioScan();
	*/
}

// This is applicable to tests that use the messaging and scheduling.
void testBiophysicsProcess()
{
	testCompartmentProcess();
	testHHChannel();
}

#endif
