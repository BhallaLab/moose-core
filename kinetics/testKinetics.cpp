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

#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"
#include "ReadKkit.h"

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
	vector< unsigned int > dims( 1, 1 );
	Id nid = shell->doCreate( "Neutral", Id(), "n", dims );
	Id mid = shell->doCreate( "MathFunc", nid, "m", dims );
	MathFunc* math = reinterpret_cast< MathFunc* >( mid.eref().data() );
	double tolerance = 1.0;

	// Element* n = nid();
	Element* m = mid();
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


void testKinetics()
{
	// testReadKkit(); // To go into regression tests.
	testMathFunc();
}

void testMpiKinetics( )
{
	//Need to update
// 	testMpiFibonacci();
}
