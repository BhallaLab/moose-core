/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <stdio.h>
#include <iomanip>
#include "../shell/Neutral.h"
#include "../builtins/Arith.h"
#include "../builtins/Mdouble.h"
#include "Dinfo.h"
#include <queue>
#include "../biophysics/Synapse.h"
#include "../biophysics/SynBase.h"
#include "../biophysics/IntFire.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "OneToOneMsg.h"
#include "../randnum/randnum.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickMgr.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

#include "../shell/Shell.h"

void showFields()
{
	const Cinfo* nc = Neutral::initCinfo();
	Id i1 = Id::nextId();
	vector< DimInfo > dims;
	Element* ret = new Element( i1, nc, "test1", dims, 1, 1 );
	// bool ret = nc->create( i1, "test1", 1 );
	assert( ret );

	// i1.eref().element()->showFields();
	cout << "." << flush;

	delete i1();
}

void testPrepackedBuffer()
{
	string arg1 = "This is arg1";
	double arg2 = 123.4;
	unsigned int arg3 = 567;
	Conv< string > conv1( arg1 );
	Conv< double > conv2( arg2 );
	Conv< unsigned int > conv3( arg3 );

	assert( conv1.size() == 2 );
	assert( conv2.size() == 1 );
	assert( conv3.size() == 1 );

	unsigned int totSize = conv1.size() + conv2.size() + conv3.size();
	double* buf = new double[ totSize ];
	double* temp = buf;

	conv1.val2buf( temp ); temp += conv1.size();
	conv2.val2buf( temp ); temp += conv2.size();
	conv3.val2buf( temp ); temp += conv3.size();

	PrepackedBuffer pb( buf, totSize );

	Conv< PrepackedBuffer > conv4( pb );

	assert( conv4.size() == pb.dataSize() + 2 );

	temp = new double[ conv4.size() ];

	unsigned int size = conv4.val2buf( temp );
	assert( size == pb.dataSize() + 2 );

	Conv< PrepackedBuffer > conv5( temp );

	PrepackedBuffer pb2 = *conv5;

	assert( pb2.dataSize() == pb.dataSize() );

	const double* temp2 = pb2.data();

	Conv< string > conv6( temp2 );
	temp2 += conv6.size();
	Conv< double > conv7( temp2 );
	temp2 += conv7.size();
	Conv< unsigned int > conv8( temp2 );
	temp2 += conv8.size();

	assert( *conv6 == arg1 );
	assert( *conv7 == arg2 );
	assert( *conv8 == arg3 );

	delete[] buf;
	delete[] temp;
	cout << "." << flush;
}

/**
 * This used to use the 'name' field of Neutral as a test variable.
 * Now no longer as useful, since the field is replaced with the 'name'
 * of the parent Element. Instead use Arith.output.
 */
void insertIntoQ( )
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	const DestFinfo* df = dynamic_cast< const DestFinfo* >(
		ac->findFinfo( "set_outputValue" ) );
	assert( df != 0 );
	FuncId fid = df->getFid();

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	DimInfo temp = { size, 1, 0 };
	vector< DimInfo > dims( 1, temp );

	Element* ret = new Element( i1, ac, "test1", dims, 1, 1 );
	assert( ret );
	ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( Msg::nextMsgId(), e1, e2 );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * i;
		// This simulates a sendTo
		Qinfo::addDirectToQ( i1, ObjId( i2, i ), 
			ScriptThreadNum, fid, &x, 1 );

		/*
		double buf[200];

		Conv< double > conv( x );
		unsigned int size = conv.val2buf( buf );
		Qinfo qi( 1, i, size + sizeof( DataId ), 1 );

		*reinterpret_cast< DataId* >( buf + size ) = DataId( i );

		MsgFuncBinding b( m->mid(), fid );

		// addToQ( threadIndex, Binding, argbuf )
		// qi.assignQblock( m, &p );
		qi.addToQforward( &p, b, buf );
		*/
	}
	Qinfo::clearQ( 0 );

	for ( unsigned int i = 0; i < size; ++i ) {
		double val = ( reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i )) )->getOutput();
		assert( doubleEq( val, i * i ) );
	}
	cout << "." << flush;

	delete m;
	delete i1();
	delete i2();
}

void testSendMsg()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	const DestFinfo* df = dynamic_cast< const DestFinfo* >(
		ac->findFinfo( "set_outputValue" ) );
	assert( df != 0 );
	FuncId fid = df->getFid();

	DimInfo temp = { size, 1, 0 };
	vector< DimInfo > dims( 1, temp );

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new Element( i1, ac, "test1", dims, 1, 1 );
	// bool ret = nc->create( i1, "test1", size );
	assert( ret );
	// ret = nc->create( i2, "test2", size );
	ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new OneToOneMsg( Msg::nextMsgId(), e1.element(), e2.element() );

	
	ThreadId threadNum = 0;
	
	SrcFinfo1<double> s( "test", "" );
	s.setBindIndex( 0 );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i + i * i;
		s.send( Eref( e1.element(), i ), threadNum, x );
	}
	Qinfo::clearQ( 0 );

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i + i * i;
		double val = reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->getOutput();
		assert( doubleEq( val, temp ) );
	}
	cout << "." << flush;

	delete i1();
	delete i2();
}

// This used to use parent/child msg, but that has other implications
// as it causes deletion of elements.
void testCreateMsg()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* temp = new Element( i1, ac, "test1", dims, 1 );
	// bool ret = nc->create( i1, "test1", size );
	assert( temp );
	temp = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( temp );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();
	ThreadId threadNum = 0;

	OneToOneMsg *m = new OneToOneMsg( Msg::nextMsgId(), e1.element(), e2.element() );
	assert( m );
	const Finfo* f1 = ac->findFinfo( "output" );
	assert( f1 );
	const Finfo* f2 = ac->findFinfo( "arg1" );
	assert( f2 );
	bool ret = f1->addMsg( f2, m->mid(), e1.element() );
	// bool ret = add( e1.element(), "child", e2.element(), "parent" );
	
	assert( ret );

	for ( unsigned int i = 0; i < size; ++i ) {
		const SrcFinfo1< double >* sf = dynamic_cast< const SrcFinfo1< double >* >( f1 );
		assert( sf != 0 );
		sf->send( Eref( e1.element(), i ), threadNum, double( i ) );
	}
	Qinfo::clearQ( 0 );

	/*
	for ( unsigned int i = 0; i < size; ++i )
		cout << i << "	" << reinterpret_cast< Neutral* >(e2.element()->data( i ))->getName() << endl;

*/
	cout << "." << flush;
	delete i1();
	delete i2();
}

/**
 * This tests the low-level Set functions, that do not involve off-node
 * messaging.
 */
void testInnerSet()
{
	// Eref sheller = Id().eref();
	// Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	DimInfo temp = { size, 1, 0 };
	vector< DimInfo > dims( 1, temp );
	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	ProcInfo p;

	Eref e2 = i2.eref();

	assert( ret->getName() == "test2" );
	const Finfo* finfo = ret->cinfo()->findFinfo( "set_name" );
	assert( finfo );
	FuncId f1 = dynamic_cast< const DestFinfo* >( finfo )->getFid();
	Conv< string > conv( "New Improved Test" );
	Qinfo::enableStructuralOps();
	Qinfo::addDirectToQ( ObjId(), ret->id(), 0, f1,
		conv.ptr(), conv.size() );
	Qinfo::clearQ( 0  );
	// Field< string >::set( e2, "name", "NewImprovedTest" );
	assert( ret->getName() == "New Improved Test" );
	

	finfo = ret->cinfo()->findFinfo( "set_outputValue" );
	assert( finfo );
	FuncId f2 = dynamic_cast< const DestFinfo* >( finfo )->getFid();

	for ( unsigned int i = 0; i < size; ++i ) {
		// double args[100];
		double x = sqrt( i );
		Conv< double > conv( x );
		Eref dest( e2.element(), i );
		Qinfo::enableStructuralOps();
		Qinfo::addDirectToQ( ObjId(), dest.objId(), 0, f2,
			conv.ptr(), conv.size() );

		Qinfo::clearQ( 0 );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = sqrt( i );
		double val = reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->getOutput();
		assert( doubleEq( val, temp ) );
	}

	cout << "." << flush;

	delete i2();
}

void testInnerGet() // Uses low-level ops to do a 'get'.
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );

	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	ProcInfo p;

	Eref e2 = i2.eref();
	const Finfo* finfo = ret->cinfo()->findFinfo( "get_name" );
	assert( finfo );
	FuncId f1 = dynamic_cast< const DestFinfo* >( finfo )->getFid();
	Qinfo q;
	Qinfo::enableStructuralOps();
	double temp = receiveGet()->getFid();
	Qinfo::addDirectToQ( ObjId(), i2, 0, f1, &temp, 1 );
	Qinfo::clearQ( 0 ); // The request goes to the target Element
	Qinfo::clearQ( 0 ); // The response comes back to the Shell
	Conv< string > conv( shell->getBuf()[0] );
	assert( *conv == "test2" );
	shell->clearGetBuf();

	ret->setName( "HupTwoThree" );
	Qinfo::enableStructuralOps();
	Qinfo::addDirectToQ( ObjId(), i2, 0, f1, &temp, 1 );

	Qinfo::clearQ( 0 ); // The request goes to the target Element
	Qinfo::clearQ( 0 ); // The response comes back to the Shell
	Conv< string > conv2( shell->getBuf()[0] );
	assert( *conv2 == "HupTwoThree" );
	shell->clearGetBuf();

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->setOutput( temp );
	}

	finfo = ret->cinfo()->findFinfo( "get_outputValue" );
	assert( finfo );
	FuncId f2 = dynamic_cast< const DestFinfo* >( finfo )->getFid();
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref dest( e2.element(), i );

		Qinfo::enableStructuralOps();
		double temp = receiveGet()->getFid();
		Qinfo::addDirectToQ( ObjId(), ObjId( i2, i ), 0, f2, &temp, 1 );
		Qinfo::clearQ( 0 ); // The request goes to the target Element
		Qinfo::clearQ( 0 ); // The response comes back to the Shell
		Conv< double > conv3( shell->getBuf()[0] );
		temp = i * 3;
		assert( doubleEq( *conv3 , temp ) );
		shell->clearGetBuf();
	}

	cout << "." << flush;
	delete i2();
}

void testSetGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );

	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	ProcInfo p;
	
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double x = i * 3.14;
		bool ret = Field< double >::set( oid, "outputValue", x );
		assert( ret );
		// Qinfo::clearQ( 0 ); // The request goes to the target Element
		double val = reinterpret_cast< Arith* >(oid.data())->getOutput();
		assert( doubleEq( val, x ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double x = i * 3.14;
		double ret = Field< double >::get( oid, "outputValue" );
		ProcInfo p;
		assert( doubleEq( ret, x ) );
	}

	cout << "." << flush;
	delete i2();
}

void testStrSet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	ProcInfo p;

	Eref e2 = i2.eref();

	assert( ret->getName() == "test2" );
	bool ok = SetGet::strSet( ObjId( i2, 0 ), "name", "NewImprovedTest" );
	assert( ok );
	assert( ret->getName() == "NewImprovedTest" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = sqrt( i );
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );
		stringstream ss;
		ss << setw( 10 ) << x;
		ok = SetGet::strSet( dest, "outputValue", ss.str() );
		assert( ok );
		// SetGet1< double >::set( dest, "set_outputValue", x );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = sqrt( i );
		double val = reinterpret_cast< Arith* >( Eref( i2(), i ).data() )->getOutput();
		assert( fabs( val - temp ) < 1e-5 );
		// DoubleEq won't work here because string is truncated.
	}

	cout << "." << flush;

	delete i2();
}

void testGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	// Element* shell = Id()();
	ProcInfo p;

	// Eref e2 = i2.eref();
	ObjId oid( i2, 0 );

	string val = Field< string >::get( oid, "name" );
	assert( val == "test2" );
	ret->setName( "HupTwoThree" );
	val = Field< string >::get( oid, "name" );
	assert( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >(oid.element()->dataHandler()->data( i ))->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );

		double val = Field< double >::get( dest, "outputValue" );
		double temp = i * 3;
		assert( doubleEq( val, temp ) );
	}

	cout << "." << flush;
	delete i2();
}

void testStrGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );
	// Element* shell = Id()();
	ProcInfo p;

	// Eref e2 = i2.eref();
	ObjId oid( i2, 0 );

	string val;
	bool ok = SetGet::strGet( oid, "name", val );
	assert( ok );
	assert( val == "test2" );
	ret->setName( "HupTwoThree" );
	ok = SetGet::strGet( oid, "name", val );
	assert( ok );
	assert( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >( ObjId( i2, i ).data() )->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );
		ok = SetGet::strGet( dest, "outputValue", val );
		assert( ok );
		double conv = atof( val.c_str() );
		double temp = i * 3;
		assert( fabs( conv - temp ) < 1e-5 );
		// DoubleEq won't work here because string is truncated.
	}

	cout << "." << flush;
	delete i2();
}


void testSetGetDouble()
{
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* ret = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( ret );
	ProcInfo p;

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref e2( i2(), i );
		ObjId oid( i2, i );
		double temp = i;
		bool ret = Field< double >::set( oid, "Vm", temp );
		// Qinfo::clearQ( 0 ); // The request goes to the target Element
		assert( ret );
		assert( 
			doubleEq ( reinterpret_cast< IntFire* >(oid.data())->getVm() , temp ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref e2( i2(), i );
		ObjId oid( i2, i );
		double temp = i;
		double ret = Field< double >::get( oid, "Vm" );
		assert( doubleEq( temp, ret ) );
	}

	cout << "." << flush;
	delete i2();
	delete i3();
}

void testSetGetSynapse()
{
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( temp );

	Id synId( i2.value() + 1 );
	Element* syn = synId();

	// Element should exist even if data doesn't
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" ); 

//	assert( syn->dataHandler()->data( 0 ) != 0 ); // Should give warning

	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	assert( syn->dataHandler()->localEntries() == 0 );
	// could/should use SetVec here.
	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref e2( i2(), i );
		ObjId oid( i2, i );
		bool ret = Field< unsigned int >::set( oid, "numSynapses", i );
		assert( ret );
	}
	// Qinfo::clearQ( ScriptThreadNum );
	assert( syn->dataHandler()->localEntries() == ( size * (size - 1) ) / 2 );

	/*
	** Can't run this here: the Process Loop has to be running first.
	Eref sheller = Id().eref();
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	const Finfo* f = IntFire::initCinfo()->findFinfo( "get_numSynapses" );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( f );
	assert( df );
	shell->doSyncDataHandler( i2, df->getFid() );
	*/

/*
	*/
	FieldDataHandlerBase * fdh =
		static_cast< FieldDataHandlerBase *>( syn->dataHandler() );
	// fdh->setFieldDimension( fdh->biggestFieldArraySize() );
	fdh->resize( 0, fdh->biggestFieldArraySize() );
	// fdh->syncFieldArraySize();
	assert( syn->dataHandler()->totalEntries() == 9900 );
	// cout << "NumSyn = " << syn.totalEntries() << endl;
	
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j, fdh->numFieldBits() );
			// Eref syne( syn, di );
			ObjId synoid( synId, di );
			double temp = i * 1000 + j ;
			bool ret = Field< double >::set( synoid, "delay", temp );
			// Qinfo::clearQ( ScriptThreadNum );
			assert( ret );
			assert( 
			doubleEq( reinterpret_cast< Synapse* >(synoid.data())->getDelay() , temp ) );
		}
	}
	cout << "." << flush;
	delete synId();
	delete i2();
}

void testSetGetVec()
{
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );

	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( temp );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->localEntries() == 0 );
	assert( syn->dataHandler()->totalEntries() == size * 65536 );

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
	// Qinfo::clearQ( ScriptThreadNum );

	for ( unsigned int i = 0; i < size; ++i ) {
		IntFire* fire = reinterpret_cast< IntFire* >( i2()->dataHandler()->data( i ) );
		assert( fire->getNumSynapses() == i );
	}

	vector< unsigned int > getSyn;

	Field< unsigned int >::getVec( i2, "numSynapses", getSyn );
	assert (getSyn.size() == size );
	for ( unsigned int i = 0; i < size; ++i )
		assert( getSyn[i] == i );

	unsigned int nd = syn->dataHandler()->localEntries();
	assert( nd == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << nd << endl;
	assert( fd->localEntries() == nd );
	
	assert( fd->biggestFieldArraySize() == size - 1 );
	// fd->setFieldDimension( size );
	fd->resize( 0, size );
	assert ( fd->totalEntries() == size * size );
	// Here we test setting a 2-D array with different dims on each axis.
	vector< double > delay( size * size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			// delay[ ( i << fd->numFieldBits() ) + j ] = i * 1000 + j;
			delay[ i * size + j ] = i * 1000 + j;
		}
	}

	Eref se( syn, 0 );
	ret = Field< double >::setVec( synId, "delay", delay );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j, fd->numFieldBits() );
			Eref syne( syn, di );
			double temp = i * 1000 + j ;
			double y = 
				reinterpret_cast< Synapse* >(syne.data())->getDelay();
			assert( doubleEq( y, temp ) );
		}
	}
	Eref syne( syn, DataId::any );
	vector< double > delayVec;
	Field< double >::getVec( synId, "delay", delayVec );
	assert( delayVec.size() == size * size - 1);
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			double temp = i * 1000 + j ;
			// assert( doubleEq( delayVec[ k++ ], temp ) );
			assert( doubleEq( delayVec[ i * size + j ], temp ) );
		}
	}

	cout << "." << flush;
	delete i3();
	delete i2();
}

void test2ArgSetVec()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );

	vector< double > arg1( size );
	vector< double > arg2( size );
	for ( unsigned int i = 0; i < size; ++i ) {
		arg1[i] = i;
		arg2[i] = 100 * ( 100 - i );
	}

	SetGet2< double, double >::setVec( i2, "arg1x2", arg1, arg2 );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double x = i * 100 * ( 100 - i );
		double val = reinterpret_cast< Arith* >(oid.data())->getOutput();
		assert( doubleEq( val, x ) );
	}
	cout << "." << flush;
	delete i2();
}

void testFastGet()
{
	const Cinfo* ic = IntFire::initCinfo();
	const Finfo* gf = ic->findFinfo("spike");
	const Finfo* vmf = ic->findFinfo("get_Vm");
	assert( gf );
	const SrcFinfo* sgf = dynamic_cast< const SrcFinfo* >( gf );
	assert( sgf );

	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id srcId = Id::nextId();
	Element* src = new Element( srcId, ic, "src", dims, 1, 1 );
	Id tgtId = Id::nextId();
	Element* tgt = new Element( tgtId, ic, "tgt", dims, 1, 1 );

	for ( unsigned int i = 0; i < size; ++i ) {
		IntFire* f = reinterpret_cast< IntFire* >( tgt->dataHandler()->data( i ) );
		f->setVm( i * i );
	}

	Msg* m = new OneToOneMsg( Msg::nextMsgId(), src, tgt );
	assert( m );
	assert( sgf->addMsg( vmf, m->mid(), src ) );

	const vector< MsgFuncBinding >* mfb = 
		src->getMsgAndFunc( sgf->getBindIndex() );
	assert( mfb->size() > 0 );

	FuncId fid = (*mfb)[0].fid;
	MsgId mid = (*mfb)[0].mid;
	assert (m->mid() == mid );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref srcer( src, i );
		double v = Field< double >::fastGet( srcer, m->mid(), fid );
		assert( doubleEq( v, i * i ) );
	}

	dims[0].size = 1;
	Id tgtId2 = Id::nextId();
	Element* tgt2 = new Element( tgtId2, ic, "tgt2", dims, 1, 1 );
	assert( tgt2 );
	Id synId( tgtId2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );
	bool ret = Field< unsigned int >::set( ObjId( tgtId2, 0 ), "numSynapses", size );
	assert( ret );
	assert( syn->dataHandler()->getFieldArraySize( 0 ) == size );
	FieldDataHandlerBase* fdh = dynamic_cast< FieldDataHandlerBase* >(
		syn->dataHandler() );
	assert( fdh );
	/*
	assert( fdh->getFieldArraySize( DataId( 0 ) ) == size );
	*/

	for ( unsigned int i = 0; i < size; ++i ) {
		DataId di( 0, i, fdh->numFieldBits() );
		Synapse* f = reinterpret_cast< Synapse* >( 
			syn->dataHandler()->data( di ) );
		f->setWeight( 5 + 10 * i - i * i );
	}

	/*
	* The OneToOneMsg between regular and field entries is undefined.
	const Cinfo* sc = Synapse::initCinfo();
	const Finfo* wtf = sc->findFinfo("get_weight");
	Msg* m2 = new OneToOneMsg( Msg::nextMsgId(), src, syn );
	assert( m2 );
	assert( sgf->addMsg( wtf, m2->mid(), src ) );

	mfb = src->getMsgAndFunc( sgf->getBindIndex() );
	assert( mfb->size() == 2 );

	fid = (*mfb)[1].fid;
	mid = (*mfb)[1].mid;
	assert (m2->mid() == mid );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref srcer( src, i );
		double v = Field< double >::fastGet( srcer, m2->mid(), fid );
		assert( doubleEq( v, 5 + 10 * i - i * i ) );
	}
	*/


	Id synId0( srcId.value() + 1 );
	delete synId0();
	Id synId1( tgtId.value() + 1 );
	delete synId1();
	Id synId2( tgtId2.value() + 1 );
	delete synId2();
	delete srcId();
	delete tgtId();
	delete tgtId2();
	cout << "." << flush;
}

class TestId {
	public :
		void setId( Id id ) {
			id_ = id;
		}
		Id getId() const {
			return id_;
		}
		static const Cinfo* initCinfo();
	private :
		Id id_ ;
};
// Here we test setRepeat using an Id field. This test is added
// because of a memory leak problem that cropped up much later.
const Cinfo* TestId::initCinfo()
{
		static ValueFinfo< TestId, Id > id(
			"id",
			"test",
			&TestId::setId,
			&TestId::getId
		);
		static Finfo* testIdFinfos[] = {&id};
		static Cinfo testIdCinfo(
			"TestIdRepeatAssignment",
			Neutral::initCinfo(),
			testIdFinfos,
			sizeof( testIdFinfos )/ sizeof( Finfo* ),
			new Dinfo< TestId >()
		);
		return &testIdCinfo;
}

void testSetRepeat()
{
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( temp );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	assert( syn->dataHandler()->localEntries() == 0 );
	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( i2, "numSynapses", numSyn );
	assert( ret );
	FieldDataHandlerBase* fdh = dynamic_cast< FieldDataHandlerBase* >( 
		syn->dataHandler() );
	assert( fdh );
	unsigned int biggestSize = fdh->biggestFieldArraySize();
	assert( biggestSize == size - 1 );
	fdh->setMaxFieldEntries( biggestSize );
	assert( syn->dataHandler()->totalEntries() == size * biggestSize );
	unsigned int nd = syn->dataHandler()->localEntries();
	assert( nd == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << nd << endl;
	
	// Here we test setting a 2-D array with different dims on each axis.
	Eref se( syn, 0 );
	ret = Field< double >::setRepeat( synId, "delay", 123.0 );
	assert( ret );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j, fdh->numFieldBits() );
			Eref syne( syn, di );
			assert( doubleEq( 
				reinterpret_cast< Synapse* >(syne.data())->getDelay(), 
				123.0 ) );
		}
	}

	Id tid = Id::nextId();
	Element* tel = new Element( tid, TestId::initCinfo(), "tid", dims, 1, 1 );
	assert( tel );
	ret = Field< Id >::setRepeat( tid, "id", i2 );
	assert( ret );

	cout << "." << flush;
	delete tid();
	delete i3();
	delete i2();
}

void testSendSpike()
{
	static const double WEIGHT = -1.0;
	static const double TAU = 1.0;
	static const double DT = 0.1;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	DimInfo dimEntry = { size, 1, 0 };
	vector< DimInfo > dims( 1, dimEntry );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
//	bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( temp );
	Eref e2 = i2.eref();
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	assert( syn->dataHandler()->localEntries() == 0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref er( i2(), i );
		ObjId oid( i2, i );
		bool ret = Field< unsigned int >::set( oid, "numSynapses", i );
		assert( ret );
	}
	FieldDataHandlerBase * fdh =
		static_cast< FieldDataHandlerBase *>( syn->dataHandler() );
	// fdh->setFieldDimension( fdh->biggestFieldArraySize() );
	fdh->resize( 0, fdh->biggestFieldArraySize() );
	// fdh->syncFieldArraySize();
	assert( fdh->localEntries() == ( size * (size - 1) ) / 2 );
	assert( fdh->totalEntries() == size * ( size - 1) );

	DataId di( 1, 0, fdh->numFieldBits() ); // DataId( data, field )
	Eref syne( syn, di );
	reinterpret_cast< Synapse* >(syne.data())->setWeight( WEIGHT );

	SingleMsg *m = new SingleMsg( Msg::nextMsgId(), e2, syne );
	const Finfo* f1 = ic->findFinfo( "spike" );
	const Finfo* f2 = Synapse::initCinfo()->findFinfo( "addSpike" );
	bool ret = f1->addMsg( f2, m->mid(), e2.element() );
	// bool ret = SingleMsg::add( e2, "spike", syne, "addSpike" );
	assert( ret );

	reinterpret_cast< IntFire* >(e2.data())->setVm( 1.0 );
	// ret = SetGet1< double >::set( e2, "Vm", 1.0 );
	ProcInfo p;
	p.dt = DT;
	reinterpret_cast< IntFire* >(e2.data())->process( e2, &p );
	// At this stage we have sent the spike, so e2.data::Vm should be -1e-7.
	double Vm = reinterpret_cast< IntFire* >(e2.data())->getVm();
	assert( doubleEq( Vm, -1e-7 ) );
	// Test that the spike message is in the queue.
	assert( Qinfo::qBuf_.size() > 0 );
	assert( Qinfo::dBuf_.size() > 0 );
	assert( Qinfo::qBuf_[0].size() == 1 );
	assert( Qinfo::dBuf_[0].size() == 1 );

	Qinfo::clearQ( 0 );
	assert( Qinfo::qBuf_[0].size() == 0 );
	assert( Qinfo::dBuf_[0].size() == 0 );

	/*
	// Warning: the 'get' function calls clearQ also.
	Vm = SetGet1< double >::get( e2, "Vm" );
	assert( fabs( Vm + 1e-7) < EPSILON );
	*/

	// Eref synParent( e2.element(), 1 );
	ObjId synParent( i2, 1 );
	reinterpret_cast< IntFire* >(synParent.data())->setTau( TAU );

	reinterpret_cast< IntFire* >(synParent.data())->process( synParent.eref(), &p );
	Vm = Field< double >::get( synParent, "Vm" );
	assert( doubleEq( Vm , WEIGHT * ( 1.0 - DT / TAU ) ) );
	// cout << "Vm = " << Vm << endl;
	cout << "." << flush;
	delete i3();
	delete i2();
}

void printSparseMatrix( const SparseMatrix< unsigned int >& m)
{
	unsigned int nRows = m.nRows();
	unsigned int nCols = m.nColumns();
	
	for ( unsigned int i = 0; i < nRows; ++i ) {
		cout << "[	";
		for ( unsigned int j = 0; j < nCols; ++j ) {
			cout << m.get( i, j ) << "	";
		}
		cout << "]\n";
	}
	const unsigned int *n;
	const unsigned int *c;
	for ( unsigned int i = 0; i < nRows; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j )
			cout << n[j] << "	";
	}
	cout << endl;

	for ( unsigned int i = 0; i < nRows; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j )
			cout << c[j] << "	";
	}
	cout << endl;
	cout << endl;
}

void testSparseMatrix()
{
	static unsigned int preN[] = { 1, 2, 3, 4, 5, 6, 7 };
	static unsigned int postN[] = { 1, 3, 4, 5, 6, 2, 7 };
	static unsigned int preColIndex[] = { 0, 4, 0, 1, 2, 3, 4 };
	static unsigned int postColIndex[] = { 0, 1, 1, 1, 2, 0, 2 };

	SparseMatrix< unsigned int > m( 3, 5 );
	unsigned int nRows = m.nRows();
	unsigned int nCols = m.nColumns();

	m.set( 0, 0, 1 );
	m.set( 0, 4, 2 );
	m.set( 1, 0, 3 );
	m.set( 1, 1, 4 );
	m.set( 1, 2, 5 );
	m.set( 2, 3, 6 );
	m.set( 2, 4, 7 );

	const unsigned int *n;
	const unsigned int *c;
	unsigned int k = 0;
	for ( unsigned int i = 0; i < nRows; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j ) {
			assert( n[j] == preN[ k ] );
			assert( c[j] == preColIndex[ k ] );
			k++;
		}
	}
	assert( k == 7 );

	// printSparseMatrix( m );

	m.transpose();

	k = 0;
	for ( unsigned int i = 0; i < nCols; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j ) {
			assert( n[j] == postN[ k ] );
			assert( c[j] == postColIndex[ k ] );
			k++;
		}
	}
	assert( k == 7 );

	cout << "." << flush;
}

void testSparseMatrix2()
{
	// Here zeroes mean no entry, not an entry of zero.
	// Rows 0 to 4 are totally empty
	static unsigned int row5[] = { 1, 0, 2, 0, 0, 0, 0, 0, 0, 0 };
	static unsigned int row6[] = { 0, 0, 3, 4, 0, 0, 0, 0, 0, 0 };
	static unsigned int row7[] = { 0, 0, 0, 0, 5, 0, 0, 0, 0, 6 };
	static unsigned int row8[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	static unsigned int row9[] = { 0, 0, 7, 0, 0, 0, 0, 8, 0, 0 };

	vector< vector < unsigned int > > m( 10 );
	for ( unsigned int i = 0; i < 10; ++i )
		m[i].resize( 10, 0 );

	for ( unsigned int i = 0; i < 10; ++i ) m[5][i] = row5[i];
	for ( unsigned int i = 0; i < 10; ++i ) m[6][i] = row6[i];
	for ( unsigned int i = 0; i < 10; ++i ) m[7][i] = row7[i];
	for ( unsigned int i = 0; i < 10; ++i ) m[8][i] = row8[i];
	for ( unsigned int i = 0; i < 10; ++i ) m[9][i] = row9[i];

	SparseMatrix< unsigned int > n( 10, 10 );
	for ( unsigned int i = 0; i < 10; ++i )
		for ( unsigned int j = 0; j < 10; ++j )
			if ( m[i][j] != 0 )
				n.set( i, j, m[i][j] );
				
	n.transpose();
	for ( unsigned int i = 0; i < 10; ++i )
		for ( unsigned int j = 0; j < 10; ++j )
			assert (n.get( j, i ) ==  m[i][j] );
	n.transpose();
	for ( unsigned int i = 0; i < 10; ++i )
		for ( unsigned int j = 0; j < 10; ++j )
			assert (n.get( i, j ) ==  m[i][j] );
	/*
	n.printInternal();
	cout << "before transpose\n";
	n.print();
	n.transpose();
	cout << "after transpose\n";
	n.print();
	n.transpose();
	cout << "after transpose back\n";
	n.print();
	*/

	cout << "." << flush;
}

/** 
 * Need to recast this for the intact version of sparse matrix 
 * balancing.
 */
void testSparseMatrixBalance()
{
/*
	SparseMatrix< unsigned int > m( 3, 6 );
	unsigned int nRows = m.nRows();
	unsigned int nCols = m.nColumns();

	for ( unsigned int i = 0; i < nRows; ++i ) {
		for ( unsigned int j = 0; j < nCols; ++j ) {
			m.set( i, j, 100 * i + j );
		}
	}

	// printSparseMatrix( m );
	sparseMatrixBalance( 2, m );
	// printSparseMatrix( m );
	
	for ( unsigned int i = 0; i < nRows; ++i ) {
		unsigned int threadNum = i % 2;
		for ( unsigned int j = 0; j < nCols; ++j ) {
			if ( ( 2 * j ) / nCols == threadNum )
				assert( m.get( i, j ) ==  100 * ( i / 2 ) + j );
			else
				assert( m.get( i, j ) ==  0 );
		}
	}

	cout << "." << flush;
	*/
}

void printGrid( Element* e, const string& field, double min, double max )
{
	static string icon = " .oO@";
	unsigned int yside = sqrt( double ( e->dataHandler()->totalEntries() ) );
	unsigned int xside = e->dataHandler()->totalEntries() / yside;
	if ( e->dataHandler()->totalEntries() % yside > 0 )
		xside++;
	
	for ( unsigned int i = 0; i < e->dataHandler()->totalEntries(); ++i ) {
		if ( ( i % xside ) == 0 )
			cout << endl;
		Eref er( e, i );
		ObjId oid( e->id(), i );
		double Vm = Field< double >::get( oid, field );
		int shape = 5.0 * ( Vm - min ) / ( max - min );
		if ( shape > 4 )
			shape = 4;
		if ( shape < 0 )
			shape = 0;
		cout << icon[ shape ];
	}
	cout << endl;
}


void testSparseMsg()
{
	static const unsigned int qSize[] =
		{ 838, 10, 6, 18, 36, 84, 150, 196, 258, 302 };
	static const unsigned int NUMSYN = 104576;
	static const double thresh = 0.2;
	static const double Vmax = 1.0;
	static const double refractoryPeriod = 0.4;
	static const double weightMax = 0.02;
	static const double delayMax = 4;
	static const double timestep = 0.2;
	static const double connectionProbability = 0.1;
	static const unsigned int runsteps = 5;
	const Cinfo* ic = IntFire::initCinfo();
	const Finfo* procFinfo = ic->findFinfo( "process" );
	assert( procFinfo );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( procFinfo );
	assert( df );
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 1024;

	DimInfo dimEntry = { size, 1, 0 };
	vector< DimInfo > dims( 1, dimEntry );
	string arg;

	mtseed( 5489UL ); // The default value, but better to be explicit.

	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* t2 = new Element( i2, ic, "test2", dims, 1, 1 );
	assert( t2 );
	Eref e2 = i2.eref();
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->localEntries() == 0 );
	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	FieldDataHandlerBase* fd = dynamic_cast< FieldDataHandlerBase* >(
		syn->dataHandler() );
	assert( fd );

	DataId di( 1, 0, fd->numFieldBits() ); // DataId( data, field )
	Eref syne( syn, di );

	/*
	/// This old utility function is replaced with the series of funcs 
	bool ret = SparseMsg::add( e2.element(), "spike", syn, "addSpike", 
		connectionProbability );
	*/
	SparseMsg* sm = new SparseMsg( Msg::nextMsgId(), e2.element(), syn );
	assert( sm );
	const Finfo* f1 = ic->findFinfo( "spike" );
	const Finfo* f2 = Synapse::initCinfo()->findFinfo( "addSpike" );
	assert( f1 && f2 );
	f1->addMsg( f2, sm->mid(), t2 );
	sm->randomConnect( connectionProbability );
	//sm->loadBalance( 1 );
	// fd->setFieldDimension( fd->biggestFieldArraySize() );
	fd->resize( 0, fd->biggestFieldArraySize() );
	// fd->syncFieldArraySize();
	unsigned int fieldSize = fd->biggestFieldArraySize();
	// cout << "fieldSize = " << fieldSize << endl;
	// fd->setFieldDimension( fieldSize );
	fd->resize( 0, fieldSize );

	unsigned int nd = syn->dataHandler()->localEntries();
//	cout << "Num Syn = " << nd << endl;
	assert( nd == NUMSYN );

	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	bool ret = Field< double >::setVec( i2, "Vm", temp );
	assert( ret );
	/*
	for ( unsigned int i = 0; i < 40; ++i )
		cout << reinterpret_cast< IntFire* >( e2.element()->data( i ) )->getVm() << "	" << temp[i] << endl;
		*/
	temp.clear();
	temp.resize( size, thresh );
	ret = Field< double >::setVec( i2, "thresh", temp );
	assert( ret );
	temp.clear();
	temp.resize( size, refractoryPeriod );
	ret = Field< double >::setVec( i2, "refractoryPeriod", temp );
	assert( ret );

	vector< double > weight( size * fieldSize, 0.0 );
	vector< double > delay( size * fieldSize, 0.0 );
	assert( syne.element()->dataHandler()->numDimensions() == 2 );
	// unsigned int k = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		// unsigned int numSyn = syne.element()->dataHandler()->numData2( i );
		unsigned int numSyn = fd->getFieldArraySize( i );
		unsigned int k = i * fieldSize;
		for ( unsigned int j = 0; j < numSyn; ++j ) {
			weight[ k + j ] = mtrand() * weightMax;
			delay[ k + j ] = mtrand() * delayMax;
			// weight[ k ] = mtrand() * weightMax;
			// delay[ k++ ] = mtrand() * delayMax;
		}
	}
	/*
	vector< double > weight;
	weight.reserve( nd );
	vector< double > delay;
	delay.reserve( nd );
	assert( syne.element()->dataHandler()->numDimensions() == 2 );
	for ( unsigned int i = 0; i < size; ++i ) {
		// unsigned int numSyn = syne.element()->dataHandler()->numData2( i );
		unsigned int numSyn = syne.element()->dataHandler()->sizeOfDim( 1 );
		for ( unsigned int j = 0; j < numSyn; ++j ) {
			weight.push_back( mtrand() * weightMax );
			delay.push_back( mtrand() * delayMax );
		}
	}
	*/
	ret = Field< double >::setVec( synId, "weight", weight );
	assert( ret );
	ret = Field< double >::setVec( synId, "delay", delay );
	assert( ret );

	// printGrid( i2(), "Vm", 0, thresh );

	ProcInfo p;
	p.dt = timestep;
	/*
	IntFire* ifire100 = reinterpret_cast< IntFire* >( e2.element()->data( 100 ) );
	IntFire* ifire900 = reinterpret_cast< IntFire* >( e2.element()->data( 900 ) );
	*/

	for ( unsigned int i = 0; i < runsteps; ++i ) {
		p.currTime += p.dt;
		i2()->process( &p, df->getFid() );
		// unsigned int numWorkerThreads = 1;
		// unsigned int startThread = 1;
		/*
		if ( Qinfo::numSimGroup() >= 2 ) {
			numWorkerThreads = Qinfo::simGroup( 1 )->numThreads;
			startThread = Qinfo::simGroup( 1 )->startThread;
		}
		*/
		unsigned int totOutqEntries = Qinfo::qBuf_[ 0 ].size();
		// unsigned int totOutqEntries = ( *Qinfo::outQ_ )[ 0 ].totalNumEntries();
		assert( totOutqEntries == qSize[i] );
		// cout << i << ": " << totOutqEntries / ( sizeof( Qinfo ) + sizeof( double ) ) << endl << endl ;
		// cout << p.currTime << "	" << ifire100->getVm() << "	" << ifire900->getVm() << endl;
		// cout << "T = " << p.currTime << ", Q size = " << Qinfo::q_[0].size() << endl;
		Qinfo::clearQ( 0 );
//		i2()->process( &p );
		// printGrid( i2(), "Vm", 0, thresh );
		// sleep(1);
	}
	// printGrid( i2(), "Vm", 0, thresh );
	
	cout << "." << flush;
	delete i3();
	delete i2();
}

void testUpValue()
{
	const Cinfo* cc = Clock::initCinfo();
	// const Cinfo* tc = Tick::initCinfo();
	unsigned int size = 10;

	// DimInfo di = { 1, 1, 0 };
	// vector< DimInfo > dims( 1, di );
	vector< DimInfo > dims;
	Id clock = Id::nextId();
	// bool ret = cc->create( clock, "clock", 1 );
	Element* temp = new Element( clock, cc, "clock", dims, 1, 1 );
	assert( temp );

	Eref clocker = clock.eref();
	Id tickId( clock.value() + 1 );
	Element* ticke = tickId();
	assert ( ticke != 0 );
	assert ( ticke->getName() == "tick" );

	assert( ticke->dataHandler()->localEntries() == 10 );
	assert( ticke->dataHandler()->totalEntries() == 16 );
	FieldDataHandlerBase * fdh =
		static_cast< FieldDataHandlerBase *>( ticke->dataHandler() );
	// fdh->setFieldDimension( fdh->biggestFieldArraySize() );
	fdh->resize( 0, fdh->biggestFieldArraySize() );
	// fdh->syncFieldArraySize();
	assert( ticke->dataHandler()->totalEntries() == 10 );
	/*
	bool ret = Field< unsigned int >::set( clocker, "numTicks", size );
	assert( ret );
	*/


	for ( unsigned int i = 0; i < size; ++i ) {
		DataId di( 0, i, fdh->numFieldBits() ); // DataId( data, field )
		ObjId oid( tickId, di );
		double dt = i;
		bool ret = Field< double >::set( oid, "dt", dt );
		assert( ret );
		double val = Field< double >::get( oid, "localdt" );
		assert( doubleEq( dt, val ) );

		dt *= 10.0;
		ret = Field< double >::set( oid, "localdt", dt );
		assert( ret );
		val = Field< double >::get( oid, "dt" );
		assert( doubleEq( dt, val ) );
	}
	cout << "." << flush;
	delete tickId();
	delete clock();
}

/**
 * This sets up a reciprocal shared Msg in which the incoming value gets
 * appended onto the corresponding value of the target. Also, as soon
 * as any of the s1 or s2 are received, the target sends out an s0 call.
 * All this is tallied for validating the unit test.
 */

static SrcFinfo0 s0( "s0", "");
class Test
{
	public:
		Test()
			: numAcks_( 0 )
		{;}

		void process( const Eref& e, ProcPtr p )
		{;}

		void handleS0() {
			numAcks_++;
		}

		void handleS1( const Eref& e, const Qinfo* q, string s ) {
			ThreadId threadNum = 0;
			s_ = s + s_;
			s0.send( e, threadNum );
		}

		void handleS2( const Eref& e, const Qinfo* q, int i1, int i2 ) {
			ThreadId threadNum = 0;
			i1_ += 10 * i1;
			i2_ += 10 * i2;
			s0.send( e, threadNum );
		}

		static Finfo* sharedVec[ 6 ];

		static const Cinfo* initCinfo()
		{
			static SharedFinfo shared( "shared", "",
				sharedVec, sizeof( sharedVec ) / sizeof( const Finfo * ) );
			static Finfo * testFinfos[] = {
				&shared,
			};

			static Cinfo testCinfo(
				"Test",
				0,
				testFinfos,
				sizeof( testFinfos ) / sizeof( Finfo* ),
				new Dinfo< Test >()
			);
	
			return &testCinfo;
		}

		string s_;
		int i1_;
		int i2_;
		int numAcks_;
};

Finfo* Test::sharedVec[6];

void testSharedMsg()
{
	static SrcFinfo1< string > s1( "s1", "" );
	static SrcFinfo2< int, int > s2( "s2", "" );
	static DestFinfo d0( "d0", "",
		new OpFunc0< Test >( & Test::handleS0 ) );
	static DestFinfo d1( "d1", "", 
		new EpFunc1< Test, string >( &Test::handleS1 ) );
	static DestFinfo d2( "d2", "", 
		new EpFunc2< Test, int, int >( &Test::handleS2 ) );

	Test::sharedVec[0] = &s0;
	Test::sharedVec[1] = &d0;
	Test::sharedVec[2] = &s1;
	Test::sharedVec[3] = &d1;
	Test::sharedVec[4] = &s2;
	Test::sharedVec[5] = &d2;
	
	Id t1 = Id::nextId();
	Id t2 = Id::nextId();
	// bool ret = Test::initCinfo()->create( t1, "test1", 1 );

	vector< DimInfo > dims;
	Element* temp = new Element( t1, Test::initCinfo(), "test1", dims, 1, 1 );
	assert( temp );
	temp = new Element( t2, Test::initCinfo(), "test2", dims, 1, 1 );
	// ret = Test::initCinfo()->create( t2, "test2", 1 );
	assert( temp );

	// Assign initial values
	Test* tdata1 = reinterpret_cast< Test* >( t1.eref().data() );
	tdata1->s_ = "tdata1";
	tdata1->i1_ = 1;
	tdata1->i2_ = 2;

	Test* tdata2 = reinterpret_cast< Test* >( t2.eref().data() );
	tdata2->s_ = "TDATA2";
	tdata2->i1_ = 5;
	tdata2->i2_ = 6;

	// Set up message. The actual routine is in Shell.cpp, but here we
	// do it independently.
	
	const Finfo* shareFinfo = Test::initCinfo()->findFinfo( "shared" );
	assert( shareFinfo != 0 );
	Msg* m = new OneToOneMsg( Msg::nextMsgId(), t1(), t2() );
	assert( m != 0 );
	bool ret = shareFinfo->addMsg( shareFinfo, m->mid(), t1() );
	assert( ret );

	// Display stuff. Need to figure out how to unit test this.
	// t1()->showMsg();
	// t2()->showMsg();

	// Send messages
	ProcInfo p;
	ThreadId threadNum = 0;
	string arg1 = " hello ";
	s1.send( t1.eref(), threadNum, arg1 );
	s2.send( t1.eref(), threadNum, 100, 200 );

	Qinfo::clearQ( 0 );
	Qinfo::clearQ( 0 );

	string arg2 = " goodbye ";
	s1.send( t2.eref(), threadNum, arg2 );
	s2.send( t2.eref(), threadNum, 500, 600 );

	Qinfo::clearQ( 0 );
	Qinfo::clearQ( 0 );

	/*
	cout << "data1: s=" << tdata1->s_ << 
		", i1=" << tdata1->i1_ << ", i2=" << tdata1->i2_ << 
		", numAcks=" << tdata1->numAcks_ << endl;
	cout << "data2: s=" << tdata2->s_ << 
		", i1=" << tdata2->i1_ << ", i2=" << tdata2->i2_ <<
		", numAcks=" << tdata2->numAcks_ << endl;
	*/
	// Check results
	
	assert( tdata1->s_ == " goodbye tdata1" );
	assert( tdata2->s_ == " hello TDATA2" );
	assert( tdata1->i1_ == 5001  );
	assert( tdata1->i2_ == 6002  );
	assert( tdata2->i1_ == 1005  );
	assert( tdata2->i2_ == 2006  );
	assert( tdata1->numAcks_ == 2  );
	assert( tdata2->numAcks_ == 2  );
	
	t1.destroy();
	t2.destroy();

	cout << "." << flush;
}

void testConvVector()
{
	vector< unsigned int > intVec;
	for ( unsigned int i = 0; i < 5; ++i )
		intVec.push_back( i * i );
	
	double buf[500];

	Conv< vector< unsigned int > > intConv( intVec );
	assert( intConv.size() == 2 + ( intVec.size() * sizeof( unsigned int ) - 1) / sizeof( double ) );
	unsigned int ret = intConv.val2buf( buf );
	assert( ret == 4 );
	assert( buf[0] == intVec.size() );

	Conv< vector< unsigned int > > testIntConv( buf );
	assert( intConv.size() == testIntConv.size() );
	vector< unsigned int > testIntVec = *testIntConv;
	assert( intVec.size() == testIntVec.size() );
	for ( unsigned int i = 0; i < intVec.size(); ++i ) {
		assert( intVec[ i ] == testIntVec[i] );
	}

	vector< string > strVec;
	strVec.push_back( "one" );
	strVec.push_back( "two" );
	strVec.push_back( "three and more and more and more" );
	strVec.push_back( "four and yet more" );

	Conv< vector< string > > strConv( strVec );
	ret = strConv.val2buf( buf );
	assert( ret = strConv.size() );
	assert( buf[0] == 4 );
	assert( buf[1] == ret );
	Conv< vector< string > > strConvTgt( buf );
	
	vector< string > tgtStr = *strConvTgt;
	assert( tgtStr.size() == 4 );
	for ( unsigned int i = 0; i < 4; ++i )
		assert( tgtStr[i] == strVec[i] );

	cout << "." << flush;
}

void testMsgField()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 10;

	const DestFinfo* df = dynamic_cast< const DestFinfo* >(
		ac->findFinfo( "set_outputValue" ) );
	assert( df != 0 );
	FuncId fid = df->getFid();

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new Element( i1, ac, "test1", dims, 1, 1 );
	assert( ret );
	ret = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( Msg::nextMsgId(), Eref( i1(), 5 ), Eref( i2(), 3 ) );
	ProcInfo p;
	ThreadId threadNum = 0;

	Id msgElmId = m->managerId();

	Element *msgElm = msgElmId();

	assert( msgElm->getName() == "singleMsg" );

	Eref msgEr = m->manager();

	SingleMsg* sm = reinterpret_cast< SingleMsg* >( msgEr.data() );
	assert( sm );
	assert ( sm == m );
	assert( sm->getI1() == DataId( 5 ) );
	assert( sm->getI2() == DataId( 3 ) );
	
	SrcFinfo1<double> s( "test", "" );
	s.setBindIndex( 0 );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 42;
		s.send( Eref( e1.element(), i ), threadNum, x );
	}
	Qinfo::clearQ( 0 );

	// Check that regular msgs go through.
	Eref tgt3( i2(), 3 );
	Eref tgt8( i2(), 8 );
	double val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	assert( doubleEq( val, 5 * 42 ) );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	assert( doubleEq( val, 0 ) );

	// Now change I1 and I2, rerun, and check.
	sm->setI1( 9 );
	sm->setI2( 8 );
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 1000;
		s.send( Eref( e1.element(), i ), threadNum, x );
	}
	Qinfo::clearQ( 0 );
	val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	assert( doubleEq( val, 5 * 42 ) );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	assert( doubleEq( val, 9000 ) );

	cout << "." << flush;

	delete i1();
	delete i2();
}

void testSetGetExtField()
{
	const Cinfo* nc = Neutral::initCinfo();
	const Cinfo* rc = Mdouble::initCinfo();
	unsigned int size = 100;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i1 = Id::nextId();
	Id i2( i1.value() + 1 );
	Id i3( i2.value() + 1 );
	Id i4( i3.value() + 1 );
	Element* e1 = new Element( i1, nc, "test", dims, 1, 1 );
	assert( e1 );
	Shell::adopt( Id(), i1 );
	Element* e2 = new Element( i2, rc, "x", dims, 1, 1 );
	assert( e2 );
	Shell::adopt( i1, i2 );
	Element* e3 = new Element( i3, rc, "y", dims, 1, 1 );
	assert( e3 );
	Shell::adopt( i1, i3 );
	Element* e4 = new Element( i4, rc, "z", dims, 1, 1 );
	assert( e4 );
	Shell::adopt( i1, i4 );
	bool ret;

	vector< double > vec;
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId a( i1, i );
		ObjId b( i1, size - i - 1);
		// Eref a( e1, i );
		// Eref b( e1, size - i - 1 );
		double temp = i;
		ret = Field< double >::set( a, "x", temp );
		assert( ret );
		double temp2  = temp * temp;
		ret = Field< double >::set( b, "y", temp2 );
		assert( ret );
		vec.push_back( temp2 - temp );
	}

	ret = Field< double >::setVec( i1, "z", vec );
	assert( ret );

	for ( unsigned int i = 0; i < size; ++i ) {
		/*
		Eref a( e2, i );
		Eref b( e3, size - i - 1 );
		Eref c( e4, i );
		*/
		ObjId a( i2, i );
		ObjId b( i3, size - i - 1 );
		ObjId c( i4, i );
		double temp = i;
		double temp2  = temp * temp;

		double v = reinterpret_cast< Mdouble* >(a.data() )->getThis();
		assert( doubleEq( v, temp ) );

		v = reinterpret_cast< Mdouble* >(b.data() )->getThis();
		assert( doubleEq( v, temp2 ) );

		v = reinterpret_cast< Mdouble* >( c.data() )->getThis();
		assert( doubleEq( v, temp2 - temp ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref a( e1, i );
		// Eref b( e1, size - i - 1 );
		ObjId a( i1, i );
		ObjId b( i1, size - i - 1 );

		double temp = i;
		double temp2  = temp * temp;
		double ret = Field< double >::get( a, "x" );
		assert( doubleEq( temp, ret ) );
		
		ret = Field< double >::get( b, "y" );
		assert( doubleEq( temp2, ret ) );

		ret = Field< double >::get( a, "z" );
		assert( doubleEq( temp2 - temp, ret ) );
		// cout << i << "	" << ret << "	temp2 = " << temp2 << endl;
	}

	cout << "." << flush;

	/*
	* This works, but I want to avoid calling the Shell specific ops here
	*
	* Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	* s->doDelete( i1 );
	*/
	i4.destroy();
	i3.destroy();
	i2.destroy();
	i1.destroy();
}

void testLookupSetGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	Element* elm = new Element( i2, ac, "test2", dims, 1, 1 );
	assert( elm );
	ObjId obj( i2, 0 );

	Arith* arith = reinterpret_cast< Arith* >(obj.data() );
	for ( unsigned int i = 0; i < 4; ++i )
		arith->setIdentifiedArg( i, 0 );
	for ( unsigned int i = 0; i < 4; ++i )
		assert( doubleEq( 0, arith->getIdentifiedArg( i ) ) );

	LookupField< unsigned int, double >::set( obj, "anyValue", 0, 100 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 1, 101 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 2, 102 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 3, 103 );

	assert( doubleEq( arith->getOutput(), 100 ) );
	assert( doubleEq( arith->getArg1(), 101 ) );
	assert( doubleEq( arith->getIdentifiedArg( 2 ), 102 ) );
	assert( doubleEq( arith->getIdentifiedArg( 3 ), 103 ) );

	for ( unsigned int i = 0; i < 4; ++i )
		arith->setIdentifiedArg( i, 17 * i + 3 );

	double ret = LookupField< unsigned int, double >::get(
		obj, "anyValue", 0 );
	assert( doubleEq( ret, 3 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 1 );
	assert( doubleEq( ret, 20 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 2 );
	assert( doubleEq( ret, 37 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 3 );
	assert( doubleEq( ret, 54 ) );
	
	cout << "." << flush;
	i2.destroy();
}

void testIsA()
{
	const Cinfo* n = Neutral::initCinfo();
	const Cinfo* a = Arith::initCinfo();
	assert( a->isA( "Arith" ) );
	assert( a->isA( "Neutral" ) );
	assert( !a->isA( "Fish" ) );
	assert( !a->isA( "Synapse" ) );
	assert( !n->isA( "Arith" ) );
	assert( n->isA( "Neutral" ) );
	cout << "." << flush;
}

double* dptr( const DataHandler* dh, unsigned int i )
{
	return reinterpret_cast< double* >( dh->data( i ) );
}

void testDataCopyZero()
{
	double x = 3.14;
	char* cx = reinterpret_cast< char* >( &x );
	Dinfo< double > Ddbl;
	vector< DimInfo > dims;
	ZeroDimHandler zd( &Ddbl, dims, 1, 1 );
	bool ok;
	assert ( zd.isAllocated() == 1 );
	assert( zd.totalEntries() == 1 );
	ok = zd.resize( 0, 1 ); // Does not permit
	assert( !ok );
	assert ( zd.isAllocated() == 1 );
	assert( zd.totalEntries() == 1 );
	assert( zd.data( 0 ) != 0 );
	assert( zd.numDimensions() == 0 );
	assert( zd.dims().size() == 0 );

	*dptr( &zd, 0 ) = 0.0;
	assert( doubleEq( *dptr( &zd, 0 ), 0.0 ) );

	zd.assign( cx, 1 );
	assert( doubleEq( *dptr( &zd, 0 ), x ) );

	DataHandler* czd = zd.copy( 1, 1, 1 );
	assert( dynamic_cast< ZeroDimHandler* >( czd ) != 0 );
	assert ( czd->isAllocated() == 1 );
	assert( czd->totalEntries() == 1 );
	assert( doubleEq( *dptr( czd, 0 ), x ) );

	DataHandler* czd1 = zd.copy( 1, 1, 10 );
	assert( dynamic_cast< OneDimHandler* >( czd1 ) != 0 );
	assert ( czd1->isAllocated() == 1 );
	assert( czd1->totalEntries() == 10 );
	for ( unsigned int i = 0; i < 10; ++i )
		assert( doubleEq( *dptr( czd1, i ), x ) );

	DataHandler* czd2 = zd.copy( 1, 1, 20 );
	assert( dynamic_cast< OneDimHandler* >( czd2 ) != 0 );
	assert ( czd2->isAllocated() == 1 );
	assert( czd2->totalEntries() == 20 );
	for ( unsigned int i = 0; i < 20; ++i )
		assert( doubleEq( *dptr( czd2, i ), x ) );

	delete czd;
	delete czd1;
	delete czd2;

	cout << "." << flush;
}

void testDataCopyOne()
{
	static const unsigned int SIZE = 10;
	double x = 2.718;
	double y[SIZE];
	for ( unsigned int i = 0; i < SIZE; ++i )
		y[i] = x * i;

	char* cy = reinterpret_cast< char* >( y );
	DimInfo temp = { SIZE, 1, 0 };
	vector< DimInfo > dims( 1, temp );
	Dinfo< double > Ddbl;
	OneDimHandler od( &Ddbl, dims, 1, 1 );
	bool ok;
	assert ( od.isAllocated() == 1 );
	assert( od.totalEntries() == SIZE );
	assert( od.data( 0 ) != 0 );
	assert( od.numDimensions() == 1 );
	assert( od.dims().size() == 1 );
	assert( od.dims()[0].size == SIZE );
	assert( od.sizeOfDim(0) == SIZE );
	assert( od.sizeOfDim(1) == 0 );

	*dptr( &od, 0 ) = 0.0;
	assert( doubleEq( *dptr( &od, 0 ), 0.0 ) );
	od.assign( cy, SIZE );
	for ( unsigned int i = 0; i < SIZE; ++i )
		assert( doubleEq( *dptr( &od, i ), x * i ) );

	DataHandler* cod = od.copy( 2, true, 1 );
	assert( dynamic_cast< OneDimHandler* >( cod ) != 0 );
	assert ( cod->isAllocated() == 1 );
	assert( cod->totalEntries() == SIZE );
	for ( unsigned int i = 0; i < SIZE; ++i )
		assert( doubleEq( *dptr( &od, i ), x * i ) );

	DataHandler* cod1 = od.copy( 2, true, 1 );
	ok = cod1->resize( 0, 27 );
	assert( ok );
	assert( dynamic_cast< OneDimHandler* >( cod1 ) != 0 );
	assert ( cod1->isAllocated() == 1 );
	assert( cod1->totalEntries() == 27 );
	assert( cod1->numDimensions() == 1 );
	assert( cod1->dims().size() == 1 );
	assert( cod1->dims()[0].size == 27 );
	assert( cod1->totalEntries() == 27 );
	assert( cod1->sizeOfDim(0) == 27 );
	assert( cod1->sizeOfDim(1) == 0 );
	for ( unsigned int i = 0; i < 27; ++i )
		assert( doubleEq( *dptr( cod1, i ), x * ( i % SIZE ) ) );

	DataHandler* cod2 = od.copy( 3, true, (SIZE - 1) );
	assert( dynamic_cast< TwoDimHandler* >( cod2 ) != 0 );
	assert ( cod2->isAllocated() == 1 );
	assert( cod2->totalEntries() == SIZE * (SIZE - 1) );
	assert( cod2->numDimensions() == 2 );
	assert( cod2->dims().size() == 2 );
	assert( cod2->dims()[0].size == SIZE );
	assert( cod2->dims()[1].size == SIZE-1 );
	assert( cod2->totalEntries() == SIZE * (SIZE - 1) );
	assert( cod2->sizeOfDim(0) == SIZE );
	assert( cod2->sizeOfDim(1) == SIZE-1 );
	assert( cod2->sizeOfDim(2) == 0 );
	for ( unsigned int i = 0; i < ( SIZE * (SIZE - 1) ); ++i )
		assert( doubleEq( *dptr( cod2, i ), x * ( i % SIZE ) ) );

	delete cod;
	delete cod1;
	delete cod2;
	cout << "." << flush;
}

void testDataCopyAny()
{
	static const unsigned int n0 = 5;
	static const unsigned int n1 = 4;
	static const unsigned int n2 = 3;
	unsigned int k = 0;
	double y[n1][n0];
	for ( unsigned int i = 0; i < n1; ++i )
		for ( unsigned int j = 0; j < n0; ++j )
			y[i][j] = k++;
	/*
	0	1	2	3	4
	5	6	7	8	9
	10	11	12	13	14
	15	16	17	18	19
	*/

	char* cy = reinterpret_cast< char* >( y );
	Dinfo< double > Ddbl;
	vector< DimInfo > dim;
	DimInfo temp = { n0, 1, 0 };
	dim.push_back( temp );
	temp.size = n1;
	dim.push_back( temp );
	AnyDimHandler ad( &Ddbl, dim, 1, true );
	assert ( ad.isAllocated() == 1 );
	assert( ad.totalEntries() == n0 * n1 );
	assert( ad.data( 0 ) != 0 );
	assert( ad.numDimensions() == 2 );
	assert( ad.dims().size() == 2 );
	assert( ad.dims()[0].size == n0 );
	assert( ad.dims()[1].size == n1 );
	assert( ad.sizeOfDim(0) == n0 );
	assert( ad.sizeOfDim(1) == n1 );
	assert( ad.sizeOfDim(2) == 0 );

	*dptr( &ad, 0 ) = 0.0;
	assert( doubleEq( *dptr( &ad, 0 ), 0.0 ) );
	ad.assign( cy, n0 * n1 );
	for ( unsigned int i = 0; i < n1; ++i )
		for ( unsigned int j = 0; j < n0; ++j )
			assert( doubleEq( *dptr( &ad, i * n0 + j ), i * n0 + j ) );

	DataHandler* cad = ad.copy( 1, true, 1 );
	assert( dynamic_cast< AnyDimHandler* >( cad ) != 0 );
	assert ( cad->isAllocated() == 1 );
	assert( cad->totalEntries() == n0 * n1 );
	assert( cad->numDimensions() == 2 );
	for ( unsigned int i = 0; i < n1; ++i )
		for ( unsigned int j = 0; j < n0; ++j )
			assert( doubleEq( *dptr( cad, i * n0 + j ), i * n0 + j ));

	/*
	0	1	2	3	4	0	1
	5	6	7	8	9	5	6
	10	11	12	13	14	10	11
	15	16	17	18	19	15	16

	0	1	2	3	4	5	6
	7	8	9	10	11	12	13
	14	15	16	17	18	19	20
	21	22	23	24	25	26	27
	*/

	DataHandler* cad1 = ad.copy( 2, true, 1 );
	cad1->resize( 0, 7 ); // dim0 goes from 5  to 7
	assert( dynamic_cast< AnyDimHandler* >( cad1 ) != 0 );
	assert ( cad1->isAllocated() == 1 );
	assert( cad1->totalEntries() == 28 );
	assert( cad1->numDimensions() == 2 );
	assert( cad1->dims().size() == 2 );
	assert( cad1->dims()[0].size == 7 );
	assert( cad1->dims()[1].size == n1 );
	assert( cad1->dims()[0].depth == 2 );
	assert( cad1->dims()[1].depth == 2 );
	assert( cad1->sizeOfDim(0) == 7 );
	assert( cad1->sizeOfDim(1) == n1 );
	assert( cad1->sizeOfDim(2) == 0 );
	for ( unsigned int i = 0; i < n1; ++i ) {
		for ( unsigned int j = 0; j < 7; ++j ) {
			unsigned int k = j % n0;
			double val = *dptr( cad1, i * 7 + j );
			assert( doubleEq( val , i * n0 + k ) );
			// cout << *dptr( cad1, i * 7 + j ) << "	";
		}
	}

	DataHandler* cad2 = ad.copy( 3, true, n2 );
	assert( dynamic_cast< AnyDimHandler* >( cad2 ) != 0 );
	assert ( cad2->isAllocated() == 1 );
	assert( cad2->totalEntries() == n0 * n1 * n2 );
	assert( cad2->numDimensions() == 3 );
	assert( cad2->dims().size() == 3 );
	assert( cad2->dims()[0].size == n0 );
	assert( cad2->dims()[1].size == n1 );
	assert( cad2->dims()[2].size == n2 );
	assert( cad2->dims()[0].depth == 3 );
	assert( cad2->dims()[1].depth == 4 );
	assert( cad2->dims()[2].depth == 4 );
	assert( cad2->sizeOfDim(0) == n0 );
	assert( cad2->sizeOfDim(1) == n1 );
	assert( cad2->sizeOfDim(2) == n2 );
	assert( cad2->sizeOfDim(3) == 0 );
	for ( unsigned int i = 0; i < n2; ++i ) {
		for ( unsigned int j = 0; j < n1; ++j ) {
			for ( unsigned int k = 0; k < n0; ++k ) {
				assert( doubleEq( 
					*dptr( cad2, i * n0 * n1 + j * n0 + k ), 
					j * n0 + k )
				);
			}
		}
	}

	delete cad;
	delete cad1;
	delete cad2;

	cout << "." << flush;
}

/*
class testForeach: public Arith
{
	public:
		testForeach() {
			vec_.resize( 0 );
		}

		pushBack( int i ) {
			vec_.push_back( i );
		}

		const vector< int >& vec() const {
			return vec_;
		}
	private:
		vector< int > vec_;
}
*/

class testForeach: public OpFuncDummy
{
	public:
		testForeach( vector< double >* vec )
			: vec_( vec )
		{;}
		void op( const Eref& e, const Qinfo* q, const double* buf ) const
		{
			vec_->push_back( *buf );
		}
		
		const vector< double >& vec() const {
			return *vec_;
		}
	private:
		vector< double >* vec_;
};

void testOneDimHandler()
{
	static const unsigned int SIZE = 1000;
	Dinfo< int > dummyDinfo;
	DimInfo diEntry = { SIZE, 1, 0 };
	vector< DimInfo > dims( 1, diEntry );
	OneDimHandler odh( &dummyDinfo, dims, 2, 0 );
	// nodeBalance( size, myNode, numNodes );
	// Check start and end entries.
	// Here we have SIZE entries on 10 nodes. Should be divided as
	// 0 to 99, then 100 to 199, and so on.
	// These would refer to the 'data' part of the DataId.
	odh.innerNodeBalance( SIZE, 0, 10 );
	assert( odh.start_ == 0 );
	assert( odh.end_ == 100 );

	odh.innerNodeBalance( SIZE, 1, 10 );
	assert( odh.start_ == 100 );
	assert( odh.end_ == 200 );

	odh.innerNodeBalance( SIZE, 9, 10 );
	assert( odh.start_ == 900 );
	assert( odh.end_ == SIZE );

	
	// Next, test the iterators
	vector< double > vec;
	testForeach tf( &vec );
	double arg = 12.34;
	Qinfo q;
	odh.foreach( &tf, 0, &q, &arg, 1, 0 );
	assert ( vec.size() == odh.localEntries() );
	for ( unsigned int i = 0; i < odh.localEntries(); ++i )
		assert( doubleEq( arg, vec[i] ) );
	/*

	OneDimHandler::iterator i = odh.begin();
	assert ( i.index() == 900 );
	assert( i.linearIndex() == 900 );

	OneDimHandler::iterator end = odh.end();
	assert( end.index() == SIZE );
	assert( end.linearIndex() == SIZE );

	++i;
	assert ( i.index() == 901 );
	assert( i.linearIndex() == 901 );
	*/
	
	cout << "." << flush;
}

void testFieldDataHandler()
{
	static const unsigned int SIZE = 1000;
	Dinfo< IntFire > dinfo1;
	Dinfo< Synapse > dinfo2;
	// Here we have SIZE entries on 10 nodes. Should be divided as
	// 0 to 99, then 100 to 199, and so on.
	// These would refer to the 'data' part of the DataId.

	unsigned int numNeurons = SIZE;
	unsigned int numNodes = 10;
	DimInfo diEntry = { SIZE, 1, 0 };
	vector< DimInfo > dims( 1, diEntry );

	OneDimHandler odh( &dinfo1, dims, 1, false );
	// Check start and end entries.
	// nodeBalance( size, myNode, numNodes );
	odh.innerNodeBalance( numNeurons, 3, numNodes );
	assert( odh.start_ == 300 );
	assert( odh.end_ == 400 );

	assert( odh.isAllocated() == 1 );
	assert( odh.totalEntries() == numNeurons );
	assert( odh.localEntries() == numNeurons / numNodes );
	assert( odh.isDataHere( 0 ) == 0 );
	assert( odh.isDataHere( 300 ) == 1 );
	assert( odh.isDataHere( 399 ) == 1 );
	assert( odh.isDataHere( 400 ) == 0 );
	assert( odh.isDataHere( 999 ) == 0 );

	// Next, test the iterators
	vector< double > vec;
	testForeach tf( &vec );
	double arg = 12.34;
	Qinfo q;
	odh.foreach( &tf, 0, &q, &arg, 1, 0 );
	assert ( vec.size() == odh.localEntries() );
	for ( unsigned int i = 0; i < odh.localEntries(); ++i )
		assert( doubleEq( arg, vec[i] ) );

	// Next, put in the FieldDataHandler.

	FieldDataHandler< IntFire, Synapse > fdh( &dinfo2, &odh, 
		65536,
		&IntFire::getSynapse,
		&IntFire::getNumSynapses,
		&IntFire::setNumSynapses
	);

	vector< unsigned int > numSynVec( numNeurons, 0 );	
	for ( unsigned int k = 0; k < numNeurons; ++k )
		fdh.setFieldArraySize( k, k );

	for ( unsigned int k = 0; k < numNeurons; ++k ) {
		if ( k >= odh.start_ && k < odh.end_ )
			assert( fdh.getFieldArraySize( k ) == k );
		else
			assert( fdh.getFieldArraySize( k ) == 0 );
	}

	assert( fdh.biggestFieldArraySize() == 399 );

	// fdh.setFieldDimension( 399 );
	// assert( fdh.getFieldDimension() == 399 );
	fdh.resize( 0, 399 );
	assert( fdh.sizeOfDim( 0 ) == 399 );


	/*
	const Cinfo* ic = IntFire::initCinfo();
	Id id = Id::nextId();
	vector< int > dims( 1, 1 );
	Element* elm = new Element( id, ic, "test2", dims, false );
	*/

	// Play with iterators
	/*
	vec.resize(0);
	arg = 56.78;
	fdh.foreach( &tf, 0, &q, &arg, 1, 0 );
	assert( vec.size() == fdh.localEntries() );
	for ( unsigned int i = 0; i < fdh.localEntries(); ++i )
		assert( doubleEq( arg, vec[i] ) );
	*/
	
	cout << "." << flush;
}

/**
 * Tests that a single copy of the FieldElement is made with all the
 * internal fields also copied over. This function examines the data
 * part of the copy.
 */
void testCopyFieldElementData()
{
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 10;

	DimInfo di = { size, 1, 0 };
	vector< DimInfo > dims( 1, di );
	string arg;
	Id i2 = Id::nextId();
	Element* origElm = new Element( i2, ic, "test2", dims, 1, true );
	assert( origElm );

	Id synId( i2.value() + 1 );
	Element* syn = synId();

	// Element should exist even if data doesn't
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" ); 

	// assert( syn->dataHandler()->data( 0 ) != 0 ); // Should generate warning

	assert( syn->dataHandler()->totalEntries() == size * 65536 );
	assert( syn->dataHandler()->localEntries() == 0 );
	// could/should use SetVec here.
	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref e2( i2(), i );
		ObjId oid( i2, i );
		bool ret = Field< unsigned int >::set( oid, "numSynapses", i );
		assert( ret );
	}
	assert( syn->dataHandler()->localEntries() == ( size * (size - 1) ) / 2 );

	FieldDataHandlerBase * fdh =
		static_cast< FieldDataHandlerBase *>( syn->dataHandler() );
	// fdh->setFieldDimension( fdh->biggestFieldArraySize() );
	fdh->resize( 0, fdh->biggestFieldArraySize() );
	assert( syn->dataHandler()->totalEntries() == ( size - 1 ) * size );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j, fdh->numFieldBits() );
			ObjId synoid( synId, di );
			double temp = i * 1000 + j ;
			bool ret = Field< double >::set( synoid, "delay", temp );
			assert( ret );
			assert( 
			doubleEq( reinterpret_cast< Synapse* >(synoid.data())->getDelay() , temp ) );
		}
	}

	///////////////////////////////////////////////////////////////////
	// All that was the setup. Here we do the copy.
	// Note that the FieldElements have to be explicitly copied.
	///////////////////////////////////////////////////////////////////
	Id copyId = Id::nextId();
	Element* copyElm = new Element( copyId, origElm, 1, 1, true );
	Id tempId = Id::nextId();
	Element* temp = new Element( tempId, syn, 1, 1, true );
	assert( temp != 0 );
	Shell::adopt( copyId, tempId );
	///////////////////////////////////////////////////////////////////
	Eref origEr( origElm, 0 );
	Eref copyEr( copyElm, 0 );
	assert( origElm->dataHandler() != copyElm->dataHandler() );
	vector< Id > origChildren;
	vector< Id > copyChildren;
	Neutral::children( origEr, origChildren );
	Neutral::children( copyEr, copyChildren );
	assert( origChildren.size() == copyChildren.size() );
	assert( origChildren.size() == 1 );
	assert( origChildren[0] == synId );
	Id copySynId = copyChildren[0];

	Element* copySynElm = copySynId();

	// Element should exist even if data doesn't
	assert ( copySynElm != 0 );
	assert ( copySynElm->getName() == "synapse" ); 
	// assert( syn->dataHandler()->data( 0 ) != 0 ); // Should generate warning
	// assert( copySynElm->dataHandler()->data( 0 ) != 0 );
	assert( copySynElm->dataHandler()->localEntries() == 
		(size * (size - 1)) /2 );
	assert( copySynElm->dataHandler()->totalEntries() == 
		( size * (size - 1) ) );

	fdh =
		static_cast< FieldDataHandlerBase *>( copySynElm->dataHandler() );
	// fdh->setFieldDimension( fdh->biggestFieldArraySize() );
	fdh->resize( 0, fdh->biggestFieldArraySize() );
	assert( copySynElm->dataHandler()->totalEntries() == 
		(size * (size - 1) ) );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j, fdh->numFieldBits() );
			ObjId synoid( synId, di );
			double temp = i * 1000 + j ;
			assert( 
			doubleEq( reinterpret_cast< Synapse* >(synoid.data())->getDelay() , temp ) );
		}
	}

	delete synId();
	delete i2();
	delete copySynId();
	delete copyId();

	cout << "." << flush;
}

void testFinfoFields()
{
	const Finfo* vmFinfo = IntFire::initCinfo()->findFinfo( "Vm" );
	const Finfo* synFinfo = IntFire::initCinfo()->findFinfo( "synapse" );
	const Finfo* procFinfo = IntFire::initCinfo()->findFinfo( "proc" );
	const Finfo* reinitFinfo = IntFire::initCinfo()->findFinfo( "reinit" );
	const Finfo* spikeFinfo = IntFire::initCinfo()->findFinfo( "spike" );
	const Finfo* createFinfo = Shell::initCinfo()->findFinfo( "requestCreate" );

	assert( vmFinfo->name() == "Vm" );
	assert( vmFinfo->docs() == "Membrane potential" );
	assert( vmFinfo->src().size() == 0 );
	assert( vmFinfo->dest().size() == 2 );
	assert( vmFinfo->dest()[0] == "set_Vm" );
	assert( vmFinfo->dest()[1] == "get_Vm" );
	assert( vmFinfo->type() == "double" );

	assert( synFinfo->name() == "synapse" );
	assert( synFinfo->docs() == "Sets up field Elements for synapse" );
	assert( synFinfo->src().size() == 0 );
	assert( synFinfo->dest().size() == 0 );
	// cout <<  synFinfo->type() << endl;
	assert( synFinfo->type() == typeid(Synapse).name() );

	assert( procFinfo->name() == "proc" );
	assert( procFinfo->docs() == "Shared message for process and reinit" );
	assert( procFinfo->src().size() == 0 );
	assert( procFinfo->dest().size() == 2 );
	assert( procFinfo->dest()[0] == "process" );
	assert( procFinfo->dest()[1] == "reinit" );
	// cout << procFinfo->type() << endl;
	assert( procFinfo->type() == "void" );

	assert( reinitFinfo->name() == "reinit" );
	assert( reinitFinfo->docs() == "Handles reinit call" );
	assert( reinitFinfo->src().size() == 0 );
	assert( reinitFinfo->dest().size() == 0 );
	// cout << reinitFinfo->type() << endl;
	assert( reinitFinfo->type() == "void" );

	assert( spikeFinfo->name() == "spike" );
	assert( spikeFinfo->docs() == "Sends out spike events" );
	assert( spikeFinfo->src().size() == 0 );
	assert( spikeFinfo->dest().size() == 0 );
	// cout << spikeFinfo->type() << endl;
	assert( spikeFinfo->type() == "double" );

	assert( createFinfo );
	assert( createFinfo->name() == "requestCreate" );
	assert( createFinfo->type() == "string,Id,Id,string,vector<int>" );

	cout << "." << flush;
}

void testCinfoFields()
{
	assert( IntFire::initCinfo()->getDocs() == "" );
	assert( IntFire::initCinfo()->getBaseClass() == "SynBase" );

	// We have a little bit of a hack here to cast away
	// constness, due to the way the FieldElementFinfos
	// are set up.
	Cinfo *neutralCinfo = const_cast< Cinfo* >( Neutral::initCinfo() );
	assert( neutralCinfo->getNumSrcFinfo() == 1 );

	Cinfo *cinfo = const_cast< Cinfo* >( IntFire::initCinfo() );
	unsigned int nsf = neutralCinfo->getNumSrcFinfo();
	assert( nsf == 1 );
	assert( cinfo->getNumSrcFinfo() == 1 + nsf );
	assert( cinfo->getSrcFinfo( 0 + nsf ) == cinfo->findFinfo( "spike" ) );

	unsigned int ndf = neutralCinfo->getNumDestFinfo();
	assert( ndf == 19 );
	unsigned int sdf = SynBase::initCinfo()->getNumDestFinfo();
	assert( sdf == 23 );
	assert( cinfo->getNumDestFinfo() == 10 + sdf );
	assert( cinfo->getDestFinfo( 0+sdf ) == cinfo->findFinfo( "set_Vm" ) );
	assert( cinfo->getDestFinfo( 1+sdf ) == cinfo->findFinfo( "get_Vm" ) );
	assert( cinfo->getDestFinfo( 2+sdf ) == cinfo->findFinfo( "set_tau" ) );
	assert( cinfo->getDestFinfo( 3+sdf ) == cinfo->findFinfo( "get_tau" ) );

	assert( cinfo->getDestFinfo( 4+sdf ) == cinfo->findFinfo( "set_thresh" ) );
	assert( cinfo->getDestFinfo( 5+sdf ) == cinfo->findFinfo( "get_thresh" ) );
	assert( cinfo->getDestFinfo( 6+sdf ) == cinfo->findFinfo( "set_refractoryPeriod" ) );
	assert( cinfo->getDestFinfo( 7+sdf ) == cinfo->findFinfo( "get_refractoryPeriod" ) );
	assert( cinfo->getDestFinfo( 8+sdf ) == cinfo->findFinfo( "process" ) );
	assert( cinfo->getDestFinfo( 9+sdf ) == cinfo->findFinfo( "reinit" ) );

	assert( cinfo->getDestFinfo( 10+sdf )->name() == "dummy" );

	unsigned int nvf = neutralCinfo->getNumValueFinfo();
	assert( nvf == 13 );
	assert( cinfo->getNumValueFinfo() == 5 + nvf );
	assert( cinfo->getValueFinfo( 0 + nvf ) == cinfo->findFinfo( "numSynapses" ) );
	assert( cinfo->getValueFinfo( 1 + nvf ) == cinfo->findFinfo( "Vm" ) );
	assert( cinfo->getValueFinfo( 2 + nvf ) == cinfo->findFinfo( "tau" ) );

	unsigned int nlf = neutralCinfo->getNumLookupFinfo();
	assert( nlf == 0 );
	assert( cinfo->getNumLookupFinfo() == 0 + nlf );
	assert( cinfo->getLookupFinfo( 0 + nlf )->name() == "dummy");

	unsigned int nshf = neutralCinfo->getNumSharedFinfo();
	assert( nshf == 0 );
	assert( cinfo->getNumSharedFinfo() == 1 + nshf );
	assert( cinfo->getSharedFinfo( 0 + nshf ) == cinfo->findFinfo( "proc" ) );

	cout << "." << flush;
}

void testCinfoElements()
{
	Id intFireCinfoId( "/classes/IntFire" );
	const Cinfo *neutralCinfo = Neutral::initCinfo();
	unsigned int nvf = neutralCinfo->getNumValueFinfo();
	unsigned int nsf = neutralCinfo->getNumSrcFinfo();
	unsigned int ndf = neutralCinfo->getNumDestFinfo();
	unsigned int sdf = SynBase::initCinfo()->getNumDestFinfo();

	assert( intFireCinfoId != Id() );
	assert( Field< string >::get( intFireCinfoId, "name" ) == "IntFire" );
	assert( Field< string >::get( intFireCinfoId, "baseClass" ) == "SynBase" );
	assert( Field< unsigned int >::get( 
		intFireCinfoId, "num_valueFinfo" ) == 5 + nvf );
	assert( Field< unsigned int >::get( 
		intFireCinfoId, "num_srcFinfo" ) == 1 + nsf );
	assert( Field< unsigned int >::get( 
		intFireCinfoId, "num_destFinfo" ) == 14 + ndf );
	
	Id intFireSrcFinfoId( "/classes/IntFire/srcFinfo" );
	assert( intFireSrcFinfoId != Id() );
	ObjId temp( intFireSrcFinfoId, DataId( nsf ) );
	string foo = Field< string >::get( temp, "name" );
	foo = Field< string >::get( temp, "type" );
	assert( Field< string >::get( temp, "name" ) == "spike" );
	assert( Field< string >::get( temp, "type" ) == "double" );

	Id intFireDestFinfoId( "/classes/IntFire/destFinfo" );
	assert( intFireDestFinfoId != Id() );
	/*
	unsigned int ret = Field< unsigned int >::get( intFireDestFinfoId, "linearSize" );
	assert( Field< unsigned int >::get( intFireDestFinfoId, "linearSize" ) == 14 );
	*/

	temp = ObjId( intFireDestFinfoId, DataId( 7 + sdf ) );
	string str = Field< string >::get( temp, "name" );
	assert( str == "get_refractoryPeriod");
	temp = ObjId( intFireDestFinfoId, DataId( 9 + sdf ) );
	// temp.dataId.fieldu= 11;
	str = Field< string >::get( temp, "name" );
	assert( str == "reinit" );
	cout << "." << flush;
}

void testMsgSrcDestFields()
{
	//////////////////////////////////////////////////////////////
	// Setup
	//////////////////////////////////////////////////////////////
	/* This is initialized in testSharedMsg()
	static SrcFinfo1< string > s1( "s1", "" );
	static SrcFinfo2< int, int > s2( "s2", "" );
	static DestFinfo d0( "d0", "",
		new OpFunc0< Test >( & Test::handleS0 ) );
	static DestFinfo d1( "d1", "", 
		new EpFunc1< Test, string >( &Test::handleS1 ) );
	static DestFinfo d2( "d2", "", 
		new EpFunc2< Test, int, int >( &Test::handleS2 ) );

	Test::sharedVec[0] = &s0;
	Test::sharedVec[1] = &d0;
	Test::sharedVec[2] = &s1;
	Test::sharedVec[3] = &d1;
	Test::sharedVec[4] = &s2;
	Test::sharedVec[5] = &d2;
	*/
	
	Id t1 = Id::nextId();
	Id t2 = Id::nextId();
	// bool ret = Test::initCinfo()->create( t1, "test1", 1 );
	vector< DimInfo > dims;
	Element* e1 = new Element( t1, Test::initCinfo(), "test1", dims,1, 1 );
	assert( e1 );
	assert( e1 == t1() );
	Element* e2 = new Element( t2, Test::initCinfo(), "test2", dims,1, 1 );
	// ret = Test::initCinfo()->create( t2, "test2", 1 );
	assert( e2 );
	assert( e2 == t2() );

	// Set up message. The actual routine is in Shell.cpp, but here we
	// do it independently.
	const Finfo* shareFinfo = Test::initCinfo()->findFinfo( "shared" );
	assert( shareFinfo != 0 );
	Msg* m = new OneToOneMsg( Msg::nextMsgId(), t1(), t2() );
	assert( m != 0 );
	bool ret = shareFinfo->addMsg( shareFinfo, m->mid(), t1() );
	assert( ret );

	//////////////////////////////////////////////////////////////
	// Test Element::getFieldsOfOutgoingMsg
	//////////////////////////////////////////////////////////////
	vector< pair< BindIndex, FuncId > > pairs;
	e1->getFieldsOfOutgoingMsg( m->mid(), pairs );
	assert( pairs.size() == 3 );
	assert( pairs[0].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[0])->getBindIndex() );
	assert( pairs[0].second == dynamic_cast< DestFinfo* >(Test::sharedVec[1])->getFid() );

	assert( pairs[1].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[2])->getBindIndex() );
	assert( pairs[1].second == dynamic_cast< DestFinfo* >(Test::sharedVec[3])->getFid() );

	assert( pairs[2].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[4])->getBindIndex() );
	assert( pairs[2].second == dynamic_cast< DestFinfo* >(Test::sharedVec[5])->getFid() );

	e2->getFieldsOfOutgoingMsg( m->mid(), pairs );
	assert( pairs.size() == 3 );

	//////////////////////////////////////////////////////////////
	// Test Cinfo::srcFinfoName
	//////////////////////////////////////////////////////////////
	assert( Test::initCinfo()->srcFinfoName( pairs[0].first ) == "s0" );
	assert( Test::initCinfo()->srcFinfoName( pairs[1].first ) == "s1" );
	assert( Test::initCinfo()->srcFinfoName( pairs[2].first ) == "s2" );

	//////////////////////////////////////////////////////////////
	// Test Cinfo::destFinfoName
	//////////////////////////////////////////////////////////////
	assert( Test::initCinfo()->destFinfoName( pairs[0].second ) == "d0" );
	assert( Test::initCinfo()->destFinfoName( pairs[1].second ) == "d1" );
	assert( Test::initCinfo()->destFinfoName( pairs[2].second ) == "d2" );
	//////////////////////////////////////////////////////////////
	// Test Msg::getSrcFieldsOnE1 and family
	//////////////////////////////////////////////////////////////
	vector< string > fieldNames;
	fieldNames = m->getSrcFieldsOnE1();
	assert( fieldNames.size() == 3 );
	assert( fieldNames[0] == "s0" );
	assert( fieldNames[1] == "s1" );
	assert( fieldNames[2] == "s2" );

	fieldNames = m->getDestFieldsOnE2();
	assert( fieldNames.size() == 3 );
	assert( fieldNames[0] == "d0" );
	assert( fieldNames[1] == "d1" );
	assert( fieldNames[2] == "d2" );

	fieldNames = m->getSrcFieldsOnE2();
	assert( fieldNames.size() == 3 );
	assert( fieldNames[0] == "s0" );
	assert( fieldNames[1] == "s1" );
	assert( fieldNames[2] == "s2" );

	fieldNames = m->getDestFieldsOnE1();
	assert( fieldNames.size() == 3 );
	assert( fieldNames[0] == "d0" );
	assert( fieldNames[1] == "d1" );
	assert( fieldNames[2] == "d2" );

	//////////////////////////////////////////////////////////////
	// Clean up.
	//////////////////////////////////////////////////////////////
	t1.destroy();
	t2.destroy();
	cout << "." << flush;
}

void testAsync( )
{
	showFields();
	testPrepackedBuffer();
	insertIntoQ();
	testSendMsg();
	testCreateMsg();
	testInnerSet();
	testInnerGet();
	testSetGet();
	testSetGetDouble();
	testSetGetSynapse();
	testSetGetVec();
	test2ArgSetVec();
	testFastGet();
	testSetRepeat();
	testStrSet();
	testStrGet();
	testLookupSetGet();
	testSendSpike();
	testSparseMatrix();
	testSparseMatrix2();
	// testSparseMatrixBalance();
	testSparseMsg();
	testUpValue();
	testSharedMsg();
	testConvVector();
	testMsgField();
	testSetGetExtField();
	testIsA();
	testDataCopyZero();
	testDataCopyOne();
	testDataCopyAny();
	testOneDimHandler();
	testFieldDataHandler();
	testCopyFieldElementData();
	testFinfoFields();
	testCinfoFields();
	testCinfoElements();
	testMsgSrcDestFields();
}
