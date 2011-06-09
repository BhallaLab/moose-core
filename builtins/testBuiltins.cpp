/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DiagonalMsg.h"
#include "OneToAllMsg.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"
#include "Arith.h"
#include "Table.h"
#include "../biophysics/Synapse.h"
#include "../biophysics/SynBase.h"
#include <queue>
#include "../biophysics/IntFire.h"

#include "../shell/Shell.h"

void testArith()
{
	Id a1id = Id::nextId();
	vector< unsigned int > dims( 1, 10 );
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims, 1 );

	Eref a1_0( a1, 0 );
	Eref a1_1( a1, 1 );

	Arith* data1_0 = reinterpret_cast< Arith* >( a1->dataHandler()->data( 0 ) );
//	Arith* data1_1 = reinterpret_cast< Arith* >( a1->data1( 1 ) );

	data1_0->arg1( 1 );
	data1_0->arg2( 0 );

	ProcInfo p;
	data1_0->process( a1_0, &p );

	assert( data1_0->getOutput() == 1 );

	data1_0->arg1( 1 );
	data1_0->arg2( 2 );

	data1_0->process( a1_0, &p );

	assert( data1_0->getOutput() == 3 );

	a1id.destroy();

	cout << "." << flush;
}

/** 
 * This test uses the Diagonal Msg and summing in the Arith element to
 * generate a Fibonacci series.
 */
void testFibonacci()
{
	if ( Shell::numNodes() > 1 )
		return;
	unsigned int numFib = 20;
	vector< unsigned int > dims( 1, numFib );

	Id a1id = Id::nextId();
	Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims );

	Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data( 0 ) );
	if ( data ) {
		data->arg1( 0 );
		data->arg2( 1 );
	}

	const Finfo* outFinfo = Arith::initCinfo()->findFinfo( "output" );
	const Finfo* arg1Finfo = Arith::initCinfo()->findFinfo( "arg1" );
	const Finfo* arg2Finfo = Arith::initCinfo()->findFinfo( "arg2" );
	const Finfo* procFinfo = Arith::initCinfo()->findFinfo( "process" );
	DiagonalMsg* dm1 = new DiagonalMsg( Msg::nextMsgId(), a1, a1 );
	bool ret = outFinfo->addMsg( arg1Finfo, dm1->mid(), a1 );
	assert( ret );
	dm1->setStride( 1 );

	DiagonalMsg* dm2 = new DiagonalMsg( Msg::nextMsgId(), a1, a1 );
	ret = outFinfo->addMsg( arg2Finfo, dm2->mid(), a1 );
	assert( ret );
	dm1->setStride( 2 );

	/*
	bool ret = DiagonalMsg::add( a1, "output", a1, "arg1", 1 );
	assert( ret );
	ret = DiagonalMsg::add( a1, "output", a1, "arg2", 2 );
	assert( ret );
	*/

	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	shell->doSetClock( 0, 1.0 );
	Eref ticker = Id( 2 ).eref();

	const Finfo* proc0Finfo = Tick::initCinfo()->findFinfo( "process0" );
	OneToAllMsg* otam = new OneToAllMsg( Msg::nextMsgId(), ticker, a1 );
	ret = proc0Finfo->addMsg( procFinfo, otam->mid(), ticker.element() );

	// ret = OneToAllMsg::add( ticker, "process0", a1, "process" );
	assert( ret );

	shell->doStart( numFib );
	unsigned int f1 = 1;
	unsigned int f2 = 0;
	for ( unsigned int i = 0; i < numFib; ++i ) {
		if ( a1->dataHandler()->isDataHere( i ) ) {
			Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data( i ) );
			// cout << Shell::myNode() << ": i = " << i << ", " << data->getOutput() << ", " << f1 << endl;
			assert( data->getOutput() == f1 );
		}
		unsigned int temp = f1;
		f1 = temp + f2;
		f2 = temp;
	}

	a1id.destroy();
	cout << "." << flush;
}

/** 
 * This test uses the Diagonal Msg and summing in the Arith element to
 * generate a Fibonacci series.
 */
void testMpiFibonacci()
{
	unsigned int numFib = 20;
	vector< unsigned int > dims( 1, numFib );

	// Id a1id = Id::nextId();
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );

	Id a1id = shell->doCreate( "Arith", Id(), "a1", dims );
	// Element* a1 = new Element( a1id, Arith::initCinfo(), "a1", dims );
	SetGet1< double >::set( a1id, "arg1", 0 );
	SetGet1< double >::set( a1id, "arg2", 1 );

	/*
	Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data( 0 ) );

	if ( data ) {
		data->arg1( 0 );
		data->arg2( 1 );
	}
	*/

	MsgId mid1 = shell->doAddMsg( "Diagonal", 
		ObjId( a1id, 0 ), "output", ObjId( a1id, 0 ), "arg1" );
	const Msg* m1 = Msg::getMsg( mid1 );
	Eref er1 = m1->manager();
	bool ret = Field< int >::set( er1.objId(), "stride", 1 );
	assert( ret );

	MsgId mid2 = shell->doAddMsg( "Diagonal", 
		ObjId( a1id, 0 ), "output", ObjId( a1id, 0 ), "arg2" );
	const Msg* m2 = Msg::getMsg( mid2 );
	Eref er2 = m2->manager();
	ret = Field< int >::set( er2.objId(), "stride", 2 );
	assert( ret );
	
	/*
	bool ret = DiagonalMsg::add( a1, "output", a1, "arg1", 1 );
	assert( ret );
	ret = DiagonalMsg::add( a1, "output", a1, "arg2", 2 );
	assert( ret );
	*/

	shell->doSetClock( 0, 1.0 );
	Eref ticker = Id( 2 ).eref();
//	ret = OneToAllMsg::add( ticker, "process0", a1, "process" );
//	assert( ret );
	shell->doUseClock( "/a1", "process", 0 );

	shell->doStart( numFib );

	vector< double > retVec;
	Field< double >::getVec( a1id, "outputValue", retVec );
	assert( retVec.size() == numFib );

	unsigned int f1 = 1;
	unsigned int f2 = 0;
	for ( unsigned int i = 0; i < numFib; ++i ) {
		/*
		if ( a1->dataHandler()->isDataHere( i ) ) {
			Arith* data = reinterpret_cast< Arith* >( a1->dataHandler()->data( i ) );
			// cout << Shell::myNode() << ": i = " << i << ", " << data->getOutput() << ", " << f1 << endl;
			assert( data->getOutput() == f1 );
		}
		*/
		assert( doubleEq( retVec[i], f1 ) );
		unsigned int temp = f1;
		f1 = temp + f2;
		f2 = temp;
	}

	a1id.destroy();
	cout << "." << flush;
}

void testUtilsForLoadXplot()
{
	bool isNamedPlot( const string& line, const string& plotname );
	double getYcolumn( const string& line );

	assert( isNamedPlot( "/plotname foo", "foo" ) );
	assert( !isNamedPlot( "/plotname foo", "bar" ) );
	assert( !isNamedPlot( "/newplot", "bar" ) );
	assert( !isNamedPlot( "", "bar" ) );
	assert( !isNamedPlot( "1234.56", "bar" ) );

	assert( doubleEq( getYcolumn( "123.456" ), 123.456 ) );
	assert( doubleEq( getYcolumn( "987	123.456" ), 123.456 ) );
	assert( doubleEq( getYcolumn( "987 23.456" ), 23.456 ) );
	assert( doubleEq( getYcolumn( "987	 3.456" ), 3.456 ) );
	assert( doubleEq( getYcolumn( "987	 0.456" ), 0.456 ) );
	assert( doubleEq( getYcolumn( "987.6	 0.456	1111.1" ), 987.6 ) );
	cout << "." << flush;
}

void testUtilsForCompareXplot()
{
	double getRMSDiff( const vector< double >& v1, const vector< double >& v2 );
	double getRMS( const vector< double >& v );

	double getRMSRatio( const vector< double >& v1, const vector< double >& v2 );

	vector< double > v1;
	vector< double > v2;
	v1.push_back( 0 );
	v1.push_back( 1 );
	v1.push_back( 2 );

	v2.push_back( 0 );
	v2.push_back( 1 );
	v2.push_back( 2 );

	double r1 = sqrt( 5.0 / 3.0 );
	double r2 = sqrt( 1.0 / 3.0 );

	assert( doubleEq( getRMS( v1 ), r1 ) );
	assert( doubleEq( getRMS( v2 ), r1 ) );
	assert( doubleEq( getRMSDiff( v1, v2 ), 0 ) );
	assert( doubleEq( getRMSRatio( v1, v2 ), 0 ) );

	v2[2] = 3;
	assert( doubleEq( getRMS( v2 ), sqrt( 10.0/3.0 ) ) );
	assert( doubleEq( getRMSDiff( v1, v2 ), r2 ) );
	assert( doubleEq( getRMSRatio( v1, v2 ), r2 / ( sqrt( 10.0/3.0 ) + r1 ) ) );
	cout << "." << flush;
}

void testTable()
{
	testUtilsForLoadXplot();
	testUtilsForCompareXplot();
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id tabid = shell->doCreate( "Table", Id(), "tab", dims );
	assert( tabid != Id() );
	Id tabentry( tabid.value() + 1 );
	Table* t = reinterpret_cast< Table* >( tabid.eref().data() );
	for ( unsigned int i = 0; i < 100; ++i ) {
		t->input( sqrt( i ) );
	}
	unsigned int numEntries = Field< unsigned int >::get( 
		tabid, "num_table" );
	assert( numEntries == 100 );
	for ( unsigned int i = 0; i < 100; ++i ) {
		ObjId temp( tabentry, DataId( 0, i ) );
		double ret = Field< double >::get( temp, "value" );
		assert( fabs( ret - sqrt( i ) ) < 1e-6 );
	}
	/*
	SetGet2< string, string >::set( 
		tabid.eref(), "xplot", "testfile", "testplot" );
		*/
	// tabentry.destroy();
	// tabid.destroy();
	shell->doDelete( tabid );
	cout << "." << flush;
}

/**
 * Tests capacity to send a request for a field value to an object
 */
void testGetMsg()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	vector< unsigned int > dims( 1, 1 );
	Id tabid = shell->doCreate( "Table", Id(), "tab", dims );
	assert( tabid != Id() );
	Id arithid = shell->doCreate( "Arith", Id(), "arith", dims );
	assert( arithid != Id() );
	// Table* t = reinterpret_cast< Table* >( tabid.eref().data() );
	MsgId ret = shell->doAddMsg( "Single", 
		tabid.eref().objId(), "requestData",
		arithid.eref().objId(), "get_outputValue" );
	assert( ret != Msg::badMsg );
	ret = shell->doAddMsg( "Single", arithid.eref().objId(), "output",
		arithid.eref().objId(), "arg1" );
	assert( ret != Msg::badMsg );
	shell->doSetClock( 0, 1 );
	shell->doUseClock( "/tab,/arith", "process", 0 );
	unsigned int numEntries = Field< unsigned int >::get( 
		tabid, "num_table" );
	assert( numEntries == 0 );
	shell->doReinit();
	SetGet1< double >::set( arithid, "arg1", 0.0 );
	SetGet1< double >::set( arithid, "arg2", 2.0 );
	shell->doStart( 100 );

	numEntries = Field< unsigned int >::get( tabid, "num_table" );
	assert( numEntries == 101 ); // One for reinit call, 100 for process.

	Id tabentry( tabid.value() + 1 );
	for ( unsigned int i = 0; i < 100; ++i ) {
		ObjId temp( tabentry, DataId( 0, i ) );
		double ret = Field< double >::get( temp, "value" );
		assert( doubleEq( ret, 2 * i ) );
	}

	// Perhaps I should do another test without reinit.
	/*
	SetGet2< string, string >::set( 
		tabid.eref(), "xplot", "testfile", "testplot" );
	tabentry.destroy();
		*/
	shell->doDelete( arithid );
	shell->doDelete( tabid );
	cout << "." << flush;
	
}

void testStatsReduce()
{
	if ( Shell::numNodes() > 1 )
		return;
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1 );
	assert( temp );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->localEntries() == 0 );
	assert( syn->dataHandler()->totalEntries() == 100 );

	FieldDataHandlerBase* fd = dynamic_cast< FieldDataHandlerBase *>( 
		syn->dataHandler() );
	assert( fd );
	assert( fd->localEntries() == 0 );

	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( i2, "numSynapses", numSyn );
	assert( ret );

	assert( fd->biggestFieldArraySize() == size - 1 );
	fd->setFieldDimension( size );
	assert ( fd->totalEntries() == size * size );
	// Here we test setting a 2-D array with different dims on each axis.
	vector< double > delay( size * size, 0.0 );
	double sum = 0.0;
	double sumsq = 0.0;
	unsigned int num = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		unsigned int k = i * size;
		for ( unsigned int j = 0; j < i; ++j ) {
			double x = i * 1000 + j;
			sum += x;
			sumsq += x * x;
			++num;
			delay[k++] = x;
		}
	}

	ret = Field< double >::setVec( synId, "delay", delay );
	Eref syner( syn, DataId::any() );

	dims[0] = 1;
	Id statsid = shell->doCreate( "Stats", Id(), "stats", dims );


	MsgId mid = shell->doAddMsg( "Reduce", 
		statsid.eref().objId(), "reduce",
		syner.objId(), "get_delay" );
	assert( mid != Msg::badMsg );
	/*
	shell->doSetClock( 0, 1 );
	shell->doReinit();
	shell->doStart( 1 );
	*/
	SetGet0::set( statsid, "trig" );
	double x = Field< double >::get( statsid, "sum" );
	assert( doubleEq( x, sum ) );
	unsigned int i = Field< unsigned int >::get( statsid, "num" );
	assert( i == num );
	x = Field< double >::get( statsid, "sdev" );
	assert( doubleEq( x, sqrt( ( sum * sum - sumsq ) /num ) ) );

	cout << "." << flush;
	delete synId();
	delete i2();
}

void testMpiStatsReduce()
{
	Shell* shell = reinterpret_cast< Shell* >( Id().eref().data() );
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	Id i2 = shell->doCreate( "IntFire", Id(), "test2", dims, 0 );
	Id synId( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->localEntries() == 0 );
	assert( syn->dataHandler()->totalEntries() == 100 );
	Eref syner( syn, 0 );

	FieldDataHandlerBase* fd = dynamic_cast< FieldDataHandlerBase *>( 
		syn->dataHandler() );
	assert( fd );
	assert( fd->localEntries() == 0 );

	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	bool ret = Field< unsigned int >::setVec( i2, "numSynapses", numSyn );
	assert( ret );

	// This calculation only works for node 0, with the present (implicit)
	// decomposition scheme.
	// assert( fd->biggestFieldArraySize() == size/Shell::numNodes() - 1 );
	Field< unsigned int >::set( synId, "fieldDimension", size );
	assert ( fd->totalEntries() == size * size );
	// Here we test setting a 2-D array with different dims on each axis.
	vector< double > delay( size * size, 0.0 );
	double sum = 0.0;
	double sumsq = 0.0;
	unsigned int num = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		unsigned int k = i * size;
		for ( unsigned int j = 0; j < i; ++j ) {
			double x = i * 1000 + j;
			sum += x;
			sumsq += x * x;
			++num;
			delay[k++] = x;
		}
	}

	ret = Field< double >::setVec( synId, "delay", delay );

	dims[0] = 1;
	Id statsid = shell->doCreate( "Stats", Id(), "stats", dims );


	MsgId mid = shell->doAddMsg( "Reduce", 
		statsid.eref().objId(), "reduce",
		syner.objId(), "get_delay" );
	assert( mid != Msg::badMsg );
	SetGet0::set( statsid, "trig" );
	double x = Field< double >::get( statsid, "sum" );
//	cout << Shell::myNode() << ": x = " << x << ", sum = " << sum << endl;
	assert( doubleEq( x, sum ) );
	unsigned int i = Field< unsigned int >::get( statsid, "num" );
	assert( i == num );
	x = Field< double >::get( statsid, "sdev" );
	assert( doubleEq( x, sqrt( ( sum * sum - sumsq ) /num ) ) );

	delete synId();
	delete i2();
	cout << "." << flush;
}

void testBuiltins()
{
	testArith();
	testTable();
}

void testBuiltinsProcess()
{
	testFibonacci();
	testGetMsg();
	testStatsReduce();
}

void testMpiBuiltins( )
{
 	testMpiFibonacci();
	testMpiStatsReduce();
}
