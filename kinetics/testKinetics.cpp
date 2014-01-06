/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MathFunc.h"

#include "../shell/Shell.h"
#include "ReadKkit.h"
#include "ReadCspace.h"
#include "EnzBase.h"
#include "MMenz.h"
#include "ReacBase.h"
#include "Reac.h"

void testReadKkit()
{
	ReadKkit rk;
	// rk.read( "test.g", "dend", 0 );
	Id base = rk.read( "foo.g", "dend", Id() );
	assert( base != Id() );
	// Id kinetics = s->doFind( "/kinetics" );

	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	rk.run();
	rk.dumpPlots( "dend.plot" );

	s->doDelete( base );
	cout << "." << flush;
}

bool isClose( double x, double y, double tol )
{
	return ( fabs( x - y ) < tol * 1e-8 );
}

void testMathFunc()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id nid = shell->doCreate( "Neutral", Id(), "n", 1 );
	Id mid = shell->doCreate( "MathFunc", nid, "m", 1 );
	MathFunc* math = reinterpret_cast< MathFunc* >( mid.eref().data() );
	double tolerance = 1.0;

	// Element* n = nid();
	Element* m = mid.element();
	assert( m != 0 );
	ProcInfo p;
	double d = 0;
	Field< string >::set( mid, "function", "f(x, y, z) = x + y*z" );
	SetGet1< double >::set( mid, "arg1", 1.0 );
	SetGet1< double >::set( mid, "arg2", 2.0 );
	SetGet1< double >::set( mid, "arg3", 3.0 );

	// SetGet1< ProcPtr >::set( mid.eref(), "process", &p );

	math->processFunc( mid.eref(), &p );

	d = Field< double >::get( mid, "result" );

	assert ( isClose( d, 7.0, tolerance ) );
	
	/*mml function*/
	
	Field< string >::set( mid, "mathML", 
		"<eq/> <apply><ci type=\"function\"> f </ci> <ci> x </ci> <ci> y </ci></apply><apply><plus/>  <ci>x</ci>  <apply><times/><ci>y</ci>  <cn>57</cn> </apply></apply> " );
	SetGet1< double >::set( mid, "arg1", 1.0 );
	SetGet1< double >::set( mid, "arg2", 2.0 );

	// SetGet1< ProcPtr >::set( mid.eref(), "process", &p );
	math->processFunc( mid.eref(), &p );
	d = Field< double >::get( mid, "result" );
	assert ( isClose( d, 115.0, tolerance ) );
	
	/*A formula from Tyson paper*/
	Field< string >::set( mid, "function", 
		"f(PP1T, CycE, CycA, CycB) = PP1T/(1 + 0.02*(2.33*(CycE+ CycA) + 1.2E1*CycB))");
	SetGet1< double >::set( mid, "arg1", 1);
	SetGet1< double >::set( mid, "arg2", 2);
	SetGet1< double >::set( mid, "arg3", 3);
	SetGet1< double >::set( mid, "arg4", 4);

	// SetGet1< ProcPtr >::set( mid.eref(), "process", &p );
	math->processFunc( mid.eref(), &p );
	// MathFunc::processFunc( &c, &p);

	d = Field< double >::get( mid, "result" );
	assert (
		isClose( d, 1/(1+0.02*(2.33*(2 + 3)+1.2E1*4)), tolerance )
	);

	//////////////////////////////////////////////////////////////////
	// Test 'op' form of MathFunc
	//////////////////////////////////////////////////////////////////
	for ( double x = 0; x < 10; x += 2.0 ) {
		vector< double > args(4);
		for ( unsigned int i = 0; i < 4; ++i )
			args[i] = i + x;
		double ret = math->op( args );
		assert( doubleEq( ret, 
			args[0]/(1+0.02*(2.33*(args[1] + args[2])+1.2E1* args[3]) ) ) );
	}
	Field< string >::set( mid, "function", 
		"f(ERG, DRG) = 0.5*(0.1*ERG + ((0.2*(DRG/0.3)^2)/(1 + (DRG/0.3)^2)))");
	SetGet1< double >::set( mid, "arg1", 1);
	SetGet1< double >::set( mid, "arg2", 2);
	// MathFunc::processFunc( &c, &p);
	math->processFunc( mid.eref(), &p );
	// SetGet1< ProcPtr >::set( mid.eref(), "process", &p );

	d = Field< double >::get( mid, "result" );
	assert (
		isClose( d, 0.5*(0.1*1 + (0.2*(2/0.3)*(2/0.3))/(1 + (2/0.3)*(2/0.3))), tolerance )
	);
	
	/*Another formula form Tyson paper*/
	/*f(ERG, DRG) = 0.5(0.1*ERG + (0.2*(DRG/0.3)^2)/(1 + (DRG/0.3)^2))*/
	Field< string >::set( mid, "mathML", 
		"<eq/> <apply> <ci type=\"function\"> f </ci> <ci> ERG </ci> <ci> DRG </ci> </apply> <apply><times/> <cn>0.5</cn> <apply><plus/> <apply><times/> <cn>0.1</cn> <ci>ERG<ci> </apply> <apply><divide/> <apply><times/> <cn>0.2</cn> <apply><power/> <apply><divide/> <ci>DRG</ci> <cn>0.3</cn> </apply> <cn>2</cn> </apply> </apply> <apply><plus/> <cn>1</cn> <apply><power/> <apply><divide/> <ci>DRG</ci> <cn>0.3</cn> </apply> <cn>2</cn> </apply> </apply> </apply> </apply> </apply>" );
	SetGet1< double >::set( mid, "arg1", 1);
	SetGet1< double >::set( mid, "arg2", 2);
	// MathFunc::processFunc( &c, &p);
// 	SetGet1< ProcPtr >::set( mid.eref(), "process", &p );
	math->processFunc( mid.eref(), &p );

	d = Field< double >::get( mid, "result" );
	//cout << d << endl;
	//cout << 0.5*(0.1*1 + (0.2*(2/0.3)*(2/0.3))/(1 + (2/0.3)*(2/0.3))) << endl;
	assert (
		isClose( d, 0.5*(0.1*1 + (0.2*(2/0.3)*(2/0.3))/(1 + (2/0.3)*(2/0.3))), tolerance )
	);
	
	mid.destroy();
	nid.destroy();
	cout << "." << flush;
}

void testMathFuncProcess()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	//////////////////////////////////////////////////////////////////////
	// This set is the test kinetic calculation using MathFunc
	//////////////////////////////////////////////////////////////////////
	Id nid = shell->doCreate( "Neutral", Id(), "n", 1 );
	Id mid = shell->doCreate( "MathFunc", nid, "m", 1 );
	Id aid = shell->doCreate( "Pool", nid, "a", 1 ); // substrate
	Id bid = shell->doCreate( "Pool", nid, "b", 1 );	// enz
	Id cid = shell->doCreate( "Pool", nid, "c", 1 ); // product
	Id tabid = shell->doCreate( "Table", nid, "tab", 1 ); // output plot

	Field< string >::set( mid, "function", "f(s, e) = (1 * e * s ) / (1 + s)" );
	Field< double >::set( aid, "nInit", 1.0 );
	Field< double >::set( bid, "nInit", 1.0 );
	Field< double >::set( cid, "nInit", 0.0 );

	shell->doAddMsg( "Single", ObjId( aid ), "nOut", ObjId( mid ), "arg1" );
	shell->doAddMsg( "Single", ObjId( bid ), "nOut", ObjId( mid ), "arg2" );
	shell->doAddMsg( "Single", ObjId( mid ), "output", ObjId( cid ), "increment" );
	shell->doAddMsg( "Single", ObjId( mid ), "output", ObjId( aid ), "decrement" );
	shell->doAddMsg( "Single", ObjId( cid ), "nOut", ObjId( tabid ), "input" );

	//////////////////////////////////////////////////////////////////////
	// This set is the reference kinetic calculation using MMEnz
	//////////////////////////////////////////////////////////////////////
	Id pid = shell->doCreate( "Pool", nid, "p", 1 ); // substrate
	Id qid = shell->doCreate( "Pool", nid, "q", 1 );	// enz
	Id rid = shell->doCreate( "Pool", nid, "r", 1 ); // product
	Id mmid = shell->doCreate( "MMenz", nid, "mm", 1 ); // product

	Id tabid2 = shell->doCreate( "Table", nid, "tab2", 1 ); //output plot

	Field< double >::set( mmid, "Km", 1.0 );
	Field< double >::set( mmid, "kcat", 1.0 );
	Field< double >::set( pid, "nInit", 1.0 );
	Field< double >::set( qid, "nInit", 1.0 );
	Field< double >::set( rid, "nInit", 0.0 );

	shell->doAddMsg( "Single", ObjId( mmid ), "sub", ObjId( pid ), "reac" );
	shell->doAddMsg( "Single", ObjId( mmid ), "prd", ObjId( rid ), "reac" );
	shell->doAddMsg( "Single", ObjId( qid ), "nOut", ObjId( mmid ), "enzDest" );
	shell->doAddMsg( "Single", ObjId( rid ), "nOut", ObjId( tabid2 ), "input" );
	shell->doSetClock( 0, 0.01 );
	shell->doUseClock( "/n/##", "process", 0 );
	
	//////////////////////////////////////////////////////////////////////
	// Now run models and compare outputs
	//////////////////////////////////////////////////////////////////////

	shell->doReinit();
	shell->doStart( 10 );

	// SetGet2< string, string >::set( tabid, "xplot", "m.out", "mathFunc" );
	// SetGet2< string, string >::set( tabid2, "xplot", "m.out", "mmenz" );
	vector< double > temp = Field< vector< double > >::get( tabid2, "vec" );
	SetGet2< vector< double >, string >::set( 
		tabid, "compareVec", temp, "rmsr" );
	double val = Field< double >::get( tabid, "outputValue" );
	assert( val < 1e-6 );

	shell->doDelete( nid );
	cout << "." << flush;
}

void testPoolVolumeScaling()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id comptId = shell->doCreate( "CylMesh", Id(), "cyl", 1 );
	Id meshId( comptId.value() + 1 );
	Id poolId = shell->doCreate( "Pool", comptId, "pool", 1 );

	ObjId mid = shell->doAddMsg( "OneToOne", 
		ObjId( poolId, 0 ), "requestVolume",
		ObjId( meshId, 0 ), "get_volume" );

	assert( mid != ObjId() );

	vector< double > coords( 9, 0.0 );
	double x1 = 100e-6;
	double r0 = 10e-6;
	double r1 = 5e-6;
	double lambda = x1;
	coords[3] = x1;
	coords[6] = r0;
	coords[7] = r1;
	coords[8] = lambda;

	Field< vector< double > >::set( comptId, "coords", coords );

	double volume = Field< double >::get( poolId, "volume" );
	assert( doubleEq( volume, PI * x1 * (r0+r1) * (r0+r1) / 4.0 ) );

	Field< double >::set( poolId, "n", 400 );
	double volscale = 1 / ( NA * volume );
	double conc = Field< double >::get( poolId, "conc" );
	assert( doubleEq( conc, 400 * volscale ) );
	Field< double >::set( poolId, "conc", 500 * volscale );
	double n = Field< double >::get( poolId, "n" );
	assert( doubleEq( n, 500 ) );

	Field< double >::set( poolId, "nInit", 650 );
	double concInit = Field< double >::get( poolId, "concInit" );
	assert( doubleEq( concInit, 650 * volscale ) );
	Field< double >::set( poolId, "concInit", 10 * volscale );
	n = Field< double >::get( poolId, "nInit" );
	assert( doubleEq( n, 10 ) );

	shell->doDelete( comptId );
	cout << "." << flush;
}

void testReacVolumeScaling()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id comptId = shell->doCreate( "CubeMesh", Id(), "cube", 1 );
	Id meshId( comptId.value() + 1 );
	Id subId = shell->doCreate( "Pool", comptId, "sub", 1 );
	Id prdId = shell->doCreate( "Pool", comptId, "prd", 1 );
	Id reacId = shell->doCreate( "Reac", comptId, "reac", 1 );

	double vol1 = 1e-15;

	ObjId mid = shell->doAddMsg( "OneToOne", 
		subId, "requestVolume", meshId, "get_volume" );
	assert( mid != ObjId() );
	mid = shell->doAddMsg( "OneToOne", 
		prdId, "requestVolume", meshId, "get_volume" );
	assert( mid != ObjId() );

	vector< double > coords( 9, 10.0e-6 );
	coords[0] = coords[1] = coords[2] = 0;

	Field< vector< double > >::set( comptId, "coords", coords );

	double volume = Field< double >::get( comptId, "volume" );
	assert( doubleEq( volume, vol1 ) );

	ObjId ret = shell->doAddMsg( "Single", reacId, "sub", subId, "reac" );
	assert( ret != ObjId() );
	ret = shell->doAddMsg( "Single", reacId, "prd", prdId, "reac" );
	assert( ret != ObjId() );

	Field< double >::set( reacId, "Kf", 2 );
	Field< double >::set( reacId, "Kb", 3 );
	double x = Field< double >::get( reacId, "kf" );
	assert( doubleEq( x, 2 ) );
	x = Field< double >::get( reacId, "kb" );
	assert( doubleEq( x, 3 ) );
	
	ret = shell->doAddMsg( "Single", reacId, "sub", subId, "reac" );
	assert( ret != ObjId() );
	double conv = 1.0 / ( NA * vol1 );
	x = Field< double >::get( reacId, "kf" );
	assert( doubleEq( x, 2 * conv ) );
	x = Field< double >::get( reacId, "kb" );
	assert( doubleEq( x, 3 ) );

	ret = shell->doAddMsg( "Single", reacId, "sub", subId, "reac" );
	assert( ret != ObjId() );
	ret = shell->doAddMsg( "Single", reacId, "prd", prdId, "reac" );
	assert( ret != ObjId() );
	x = Field< double >::get( reacId, "kf" );
	assert( doubleEq( x, 2 * conv * conv ) );
	x = Field< double >::get( reacId, "kb" );
	assert( doubleEq( x, 3 * conv ) );

	shell->doDelete( comptId );
	cout << "." << flush;
}

// See what Element::getNeighbours does with 2 sub <----> prd.
void testTwoReacGetNeighbours()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id comptId = shell->doCreate( "CubeMesh", Id(), "cube", 1 );
	Id meshId( comptId.value() + 1 );
	Id subId = shell->doCreate( "Pool", comptId, "sub", 1 );
	Id prdId = shell->doCreate( "Pool", comptId, "prd", 1 );
	Id reacId = shell->doCreate( "Reac", comptId, "reac", 1 );

	ObjId mid = shell->doAddMsg( "OneToOne", 
		subId, "requestVolume", meshId, "get_volume" );
	assert( mid != ObjId() );
	mid = shell->doAddMsg( "OneToOne", 
		prdId, "requestVolume", meshId, "get_volume" );
	assert( mid != ObjId() );

	ObjId ret = shell->doAddMsg( "Single", reacId, "sub", subId, "reac" );
	assert( ret != ObjId() );
	ret = shell->doAddMsg( "Single", reacId, "sub", subId, "reac" );
	assert( ret != ObjId() );

	ret = shell->doAddMsg( "Single", reacId, "prd", prdId, "reac" );
	assert( ret != ObjId() );

	vector< Id > pools;
	unsigned int num = reacId.element()->getNeighbours( pools, 
		Reac::initCinfo()->findFinfo( "toSub" ) );
	assert( num == 2 );
	assert( pools[0] == subId );
	assert( pools[1] == subId );

	pools.clear();
	num = reacId.element()->getNeighbours( pools, 
		Reac::initCinfo()->findFinfo( "sub" ) );
	assert( num == 2 );
	assert( pools[0] == subId );
	assert( pools[1] == subId );

	shell->doDelete( comptId );
	cout << "." << flush;
}



void testReadCspace()
{
	ReadCspace rc;
	rc.testReadModel();
	cout << "." << flush;
}

void testMMenz()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	Id mmid = shell->doCreate( "MMenz", Id(), "mm", 1 ); // mmenz
	MMenz m;
	ProcInfo p;

	m.vSetKm( mmid.eref(), 5.0 );
	m.vSetKcat( mmid.eref(), 4.0 );
	m.vReinit( mmid.eref(), &p );
	m.vSub( 2 );
	m.vEnz( 3 );
	assert( doubleEq( m.vGetKm( mmid.eref() ), 5.0 ) );
	assert( doubleEq( m.vGetKcat( mmid.eref() ), 4.0 ) );
	m.vProcess( mmid.eref(), &p );

	shell->doDelete( mmid );
	cout << "." << flush;
}

///////////////////////////////////////////////////////////////////////
// The equation for conc of substrate of an MMenz reaction is a nasty
// transcendental, so here we just work backwards and estimate t from
// the substrate concentration
///////////////////////////////////////////////////////////////////////
double estT( double s )
{
	double E = 1.0;
	double Km = 1.0;
	double kcat = 1.0;
	double s0 = 1.0;
	double c = -Km * log( s0 ) - s0;
	double t = (-1.0 /( E * kcat ) ) * ( ( Km * log( s ) + s ) + c );
	return t;
}

void testMMenzProcess()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	//////////////////////////////////////////////////////////////////////
	// This set is the test kinetic calculation using MathFunc
	//////////////////////////////////////////////////////////////////////
	Id nid = shell->doCreate( "Neutral", Id(), "n", 1 );
	//////////////////////////////////////////////////////////////////////
	// This set is the reference kinetic calculation using MMEnz
	//////////////////////////////////////////////////////////////////////
	Id pid = shell->doCreate( "Pool", nid, "p", 1 ); // substrate
	Id qid = shell->doCreate( "Pool", nid, "q", 1 );	// enz mol
	Id rid = shell->doCreate( "Pool", nid, "r", 1 ); // product
	Id mmid = shell->doCreate( "MMenz", nid, "mm", 1 ); // mmenz

	Id tabid2 = shell->doCreate( "Table", nid, "tab2", 1 ); //output plot

	Field< double >::set( mmid, "Km", 1.0 );
	Field< double >::set( mmid, "kcat", 1.0 );
	Field< double >::set( pid, "nInit", 1.0 );
	Field< double >::set( qid, "nInit", 1.0 );
	Field< double >::set( rid, "nInit", 0.0 );

	shell->doAddMsg( "Single", ObjId( mmid ), "sub", ObjId( pid ), "reac" );
	shell->doAddMsg( "Single", ObjId( mmid ), "prd", ObjId( rid ), "reac" );
	shell->doAddMsg( "Single", ObjId( qid ), "nOut", ObjId( mmid ), "enzDest" );
	shell->doAddMsg( "Single", ObjId( pid ), "nOut", ObjId( tabid2 ), "input" );
	shell->doSetClock( 0, 0.01 );
	shell->doSetClock( 1, 0.01 );
	shell->doUseClock( "/n/mm,/n/tab2", "process", 0 );
	shell->doUseClock( "/n/#[ISA=Pool]", "process", 1 );
	
	//////////////////////////////////////////////////////////////////////
	// Now run models and compare outputs
	//////////////////////////////////////////////////////////////////////

	shell->doReinit();
	shell->doStart( 10 );

	vector< double > vec = Field< vector< double > >::get( tabid2, "vec" );
	assert( vec.size() == 1001 );
	for ( unsigned int i = 0; i < vec.size(); ++i ) {
		double t = 0.01 * i;
		double et = estT( vec[i] );
		assert( doubleApprox( t, et ) );
	}

	shell->doDelete( nid );
	cout << "." << flush;
}

void testWriteKkit( Id id )
{
	extern void writeKkit( Id model, const string& s );
	writeKkit( id, "kkitWriteTest.g" );
	cout << "." << flush;
}

void testVolSort()
{
	vector< unsigned int > findVolOrder( const vector< double >& vols );
	vector< double > vols( 8 );
	vols[0] = 7;
	vols[1] = 8;
	vols[2] = 6;
	vols[3] = 5;
	vols[4] = 1;
	vols[5] = 2;
	vols[6] = 3;
	vols[7] = 4;
	vector< unsigned int > order = findVolOrder( vols );
	// The order is the rank of the volume entry, largest should be 0.
	assert( order[0] == 1 );
	assert( order[1] == 0 );
	assert( order[2] == 2 );
	assert( order[3] == 3 );
	assert( order[4] == 7 );
	assert( order[5] == 6 );
	assert( order[6] == 5 );
	assert( order[7] == 4 );

	// This is a sequence which failed in a model test, despite the
	// above test working.
	vols.resize(5);
	vols[0] = 1e-15;
	vols[1] = 3e-15;
	vols[2] = -1;
	vols[3] = 2e-15;
	vols[4] = 5e-15;
	order = findVolOrder( vols );
	assert( order[0] == 4 );
	assert( order[1] == 1 );
	assert( order[2] == 3 );
	assert( order[3] == 0 );
	assert( order[4] == 2 );
}

void testKinetics()
{
	testTwoReacGetNeighbours();
	testMMenz();
	testMathFunc();
	testPoolVolumeScaling();
	testReacVolumeScaling();
	testReadCspace();
	testVolSort();

	// This is now handled with real models in the regression tests.
	// testWriteKkit( Id() ); 
}

void testMpiKinetics( )
{
}

void testKineticsProcess( )
{
	testMMenzProcess();
	testMathFuncProcess();
}
