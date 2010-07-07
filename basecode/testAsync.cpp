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
#include "../biophysics/IntFire.h"
#include "MsgManager.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "OneToOneMsg.h"
#include "../randnum/randnum.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

void showFields()
{
	const Cinfo* nc = Neutral::initCinfo();
	Id i1 = Id::nextId();
	vector< unsigned int > dims( 1, 1 );
	Element* ret = new Element( i1, nc, "test1", dims, 1 );
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

	unsigned int totSize = conv1.size() + conv2.size() + conv3.size();
	char* buf = new char[ totSize ];
	char* temp = buf;

	conv1.val2buf( temp ); temp += conv1.size();
	conv2.val2buf( temp ); temp += conv2.size();
	conv3.val2buf( temp ); temp += conv3.size();

	PrepackedBuffer pb( buf, totSize );

	Conv< PrepackedBuffer > conv4( pb );

	assert( conv4.size() == pb.dataSize() + 2 * sizeof( unsigned int ) );

	temp = new char[ conv4.size() ];

	unsigned int size = conv4.val2buf( temp );
	assert( size == pb.dataSize() + 2 * sizeof( unsigned int ) );

	Conv< PrepackedBuffer > conv5( temp );

	PrepackedBuffer pb2 = *conv5;

	assert( pb2.dataSize() == pb.dataSize() );

	const char* temp2 = pb2.data();

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
	vector< unsigned int > dims( 1, size );

	Element* ret = new Element( i1, ac, "test1", dims, 1 );
	// bool ret = nc->create( i1, "test1", size );
	assert( ret );
	ret = new Element( i2, ac, "test2", dims, 1 );
	// ret = nc->create( i2, "test2", size );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( e1, e2 );
	ProcInfo p;

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * i;
		char buf[200];

		// This simulates a sendTo
		Conv< double > conv( x );
		unsigned int size = conv.val2buf( buf );
		Qinfo qi( 1, i, size + sizeof( DataId ), 1 );

		*reinterpret_cast< DataId* >( buf + size ) = DataId( i );

		MsgFuncBinding b( m->mid(), fid );

		// addToQ( threadIndex, Binding, argbuf )
		qi.assignQblock( m, &p );
		qi.addToQ( 0, b, buf );
	}
	Qinfo::clearQ( &p );

	for ( unsigned int i = 0; i < size; ++i ) {
		double val = ( reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i )) )->getOutput();
		assert( fabs( val - i * i ) < 1e-8 );
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
	vector< unsigned int > dims( 1, size );

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new Element( i1, ac, "test1", dims, 1 );
	// bool ret = nc->create( i1, "test1", size );
	assert( ret );
	// ret = nc->create( i2, "test2", size );
	ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new OneToOneMsg( e1.element(), e2.element() );

	
	ProcInfo p;
	
	// Defaults to BindIndex of 0.
	SrcFinfo1<double> s( "test", "" );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i + i * i;
		s.send( Eref( e1.element(), i ), &p, x );
	}
	Qinfo::clearQ( &p );

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i + i * i;
		double val = reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->getOutput();
		assert( fabs( val - temp ) < 1e-8 );
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
	vector< unsigned int > dims( 1, size );
	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* temp = new Element( i1, ac, "test1", dims, 1 );
	// bool ret = nc->create( i1, "test1", size );
	assert( temp );
	temp = new Element( i2, ac, "test2", dims, 1 );
	assert( temp );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();
	ProcInfo p;

	OneToOneMsg *m = new OneToOneMsg( e1.element(), e2.element() );
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
		sf->send( Eref( e1.element(), i ), &p, double( i ) );
	}
	Qinfo::clearQ( &p );

	/*
	for ( unsigned int i = 0; i < size; ++i )
		cout << i << "	" << reinterpret_cast< Neutral* >(e2.element()->data( i ))->getName() << endl;

*/
	cout << "." << flush;
	delete i1();
	delete i2();
}

void testSet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );
	ProcInfo p;

	Eref e2 = i2.eref();

	assert( ret->getName() == "test2" );
	Field< string >::set( e2, "name", "NewImprovedTest" );
	assert( ret->getName() == "NewImprovedTest" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = sqrt( i );
		Eref dest( e2.element(), i );
		SetGet1< double >::set( dest, "set_outputValue", x );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = sqrt( i );
		double val = reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->getOutput();
		assert( fabs( val - temp ) < 1e-6 );
	}

	cout << "." << flush;

	delete i2();
}

void testStrSet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Element* ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );
	ProcInfo p;

	Eref e2 = i2.eref();

	assert( ret->getName() == "test2" );
	bool ok = SetGet::strSet( e2, "name", "NewImprovedTest" );
	assert( ok );
	assert( ret->getName() == "NewImprovedTest" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = sqrt( i );
		Eref dest( e2.element(), i );
		stringstream ss;
		ss << setw( 7 ) << x;
		ok = SetGet::strSet( dest, "outputValue", ss.str() );
		assert( ok );
		// SetGet1< double >::set( dest, "set_outputValue", x );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = sqrt( i );
		double val = reinterpret_cast< Arith* >( Eref( i2(), i ).data() )->getOutput();
		assert( fabs( val - temp ) < 1e-5 );
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
	vector< unsigned int > dims( 1, size );
	Element* ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );
	// Element* shell = Id()();
	ProcInfo p;

	Eref e2 = i2.eref();

	string val = Field< string >::get( e2, "name" );
	assert( val == "test2" );
	ret->setName( "HupTwoThree" );
	val = Field< string >::get( e2, "name" );
	assert( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >(e2.element()->dataHandler()->data( i ))->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref dest( e2.element(), i );

		double val = Field< double >::get( dest, "outputValue" );
		double temp = i * 3;
		assert( fabs( val - temp ) < 1e-8 );
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
	vector< unsigned int > dims( 1, size );
	Element* ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );
	// Element* shell = Id()();
	ProcInfo p;

	Eref e2 = i2.eref();

	string val;
	bool ok = SetGet::strGet( e2, "name", val );
	assert( ok );
	assert( val == "test2" );
	ret->setName( "HupTwoThree" );
	ok = SetGet::strGet( e2, "name", val );
	assert( ok );
	assert( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >( Eref( i2(), i ).data() )->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref dest( e2.element(), i );
		ok = SetGet::strGet( dest, "outputValue", val );
		assert( ok );
		double conv = atof( val.c_str() );
		double temp = i * 3;
		assert( fabs( conv - temp ) < 1e-5 );
	}

	cout << "." << flush;
	delete i2();
}

void testSetGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	// bool ret = nc->create( i2, "test2", size );
	Element* ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );

	
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double x = i * 3.14;
		bool ret = Field< double >::set( e2, "outputValue", x );
		assert( ret );
		double val = reinterpret_cast< Arith* >(e2.data())->getOutput();
		assert( fabs( val - x ) < 1e-8 );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double x = i * 3.14;
		double ret = Field< double >::get( e2, "outputValue" );
		assert( fabs( ret - x ) < 1e-8 );
	}

	cout << "." << flush;
	delete i2();
}

void testSetGetDouble()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* ret = new Element( i2, ic, "test2", dims, 1 );
	assert( ret );

	// i2()->showFields();

	
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double temp = i;
		bool ret = Field< double >::set( e2, "Vm", temp );
		assert( ret );
		assert( 
			fabs ( reinterpret_cast< IntFire* >(e2.data())->getVm() - temp ) <
				EPSILON ); 
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double temp = i;
		double ret = Field< double >::get( e2, "Vm" );
		assert( fabs ( temp - ret ) < EPSILON );
	}

	cout << "." << flush;
	delete i2();
	delete i3();
}

void testSetGetSynapse()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1 );
	assert( temp );

	Id synId( i2.value() + 1 );
	Element* syn = synId();

	// Element should exist even if data doesn't
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" ); 

	assert( syn->dataHandler()->data( 0 ) == 0 );

	assert( syn->dataHandler()->numData() == 0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		bool ret = Field< unsigned int >::set( e2, "numSynapses", i );
		assert( ret );
	}
	assert( syn->dataHandler()->numData() == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << syn.numData() << endl;
	
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j );
			Eref syne( syn, di );
			double temp = i * 1000 + j ;
			bool ret = Field< double >::set( syne, "delay", temp );
			assert( ret );
			assert( 
			fabs ( reinterpret_cast< Synapse* >(syne.data())->getDelay() - temp ) <
				EPSILON ); 
		}
	}
	cout << "." << flush;
	delete i3();
	delete i2();
}

void testSetGetVec()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1 );
	assert( temp );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );
	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( e2, "numSynapses", numSyn );
	assert( ret );
	unsigned int nd = syn->dataHandler()->numData();
	assert( nd == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << nd << endl;
	
	// Here we test setting a 2-D array with different dims on each axis.
	vector< double > delay( nd, 0.0 );
	unsigned int k = 0;
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			delay[k++] = i * 1000 + j;
		}
	}

	Eref se( syn, 0 );
	ret = Field< double >::setVec( se, "delay", delay );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j );
			Eref syne( syn, di );
			double temp = i * 1000 + j ;
			assert( 
			fabs ( reinterpret_cast< Synapse* >(syne.data())->getDelay() - temp ) <
				EPSILON ); 
		}
	}
	cout << "." << flush;
	delete i3();
	delete i2();
}

void testSetRepeat()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1 );
	assert( temp );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );
	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( e2, "numSynapses", numSyn );
	assert( ret );
	unsigned int nd = syn->dataHandler()->numData();
	assert( nd == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << nd << endl;
	
	// Here we test setting a 2-D array with different dims on each axis.
	Eref se( syn, 0 );
	ret = Field< double >::setRepeat( se, "delay", 123.0 );
	assert( ret );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j );
			Eref syne( syn, di );
			assert( 
			fabs ( reinterpret_cast< Synapse* >(syne.data())->getDelay() - 123.0 ) <
				EPSILON ); 
		}
	}
	cout << "." << flush;
	delete i3();
	delete i2();
}

void testSendSpike()
{
	static const double EPSILON = 1e-9;
	static const double WEIGHT = -1.0;
	static const double TAU = 1.0;
	static const double DT = 0.1;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
//	bool ret = ic->create( i2, "test2", size );
	Element* temp = new Element( i2, ic, "test2", dims, 1 );
	assert( temp );
	Eref e2 = i2.eref();
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref er( i2(), i );
		bool ret = Field< unsigned int >::set( er, "numSynapses", i );
		assert( ret );
	}
	assert( syn->dataHandler()->numData() == ( size * (size - 1) ) / 2 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( syn, di );
	reinterpret_cast< Synapse* >(syne.data())->setWeight( WEIGHT );

	SingleMsg *m = new SingleMsg( e2, syne );
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
	assert( fabs( Vm + 1e-7) < EPSILON );
	// Test that the spike message is in the queue.
	assert( Qinfo::outQ_[0].size() == sizeof( Qinfo ) + sizeof( double ) );

	Qinfo::clearQ( &p );
	assert( Qinfo::outQ_[0].size() == 0 );

	/*
	// Warning: the 'get' function calls clearQ also.
	Vm = SetGet1< double >::get( e2, "Vm" );
	assert( fabs( Vm + 1e-7) < EPSILON );
	*/

	Eref synParent( e2.element(), 1 );
	reinterpret_cast< IntFire* >(synParent.data())->setTau( TAU );

	reinterpret_cast< IntFire* >(synParent.data())->process( synParent, &p );
	Vm = Field< double >::get( synParent, "Vm" );
	assert( fabs( Vm - WEIGHT * ( 1.0 - DT / TAU ) ) < EPSILON );
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

void testSparseMatrixBalance()
{
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
}

void printGrid( Element* e, const string& field, double min, double max )
{
	static string icon = " .oO@";
	unsigned int yside = sqrt( double ( e->dataHandler()->numData() ) );
	unsigned int xside = e->dataHandler()->numData() / yside;
	if ( e->dataHandler()->numData() % yside > 0 )
		xside++;
	
	for ( unsigned int i = 0; i < e->dataHandler()->numData(); ++i ) {
		if ( ( i % xside ) == 0 )
			cout << endl;
		Eref er( e, i );
		double Vm = Field< double >::get( er, field );
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
	vector< unsigned int > dims( 1, size );
	string arg;

	mtseed( 5489UL ); // The default value, but better to be explicit.

	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	// bool ret = ic->create( i2, "test2", size );
	Element* t2 = new Element( i2, ic, "test2", dims, 1 );
	assert( t2 );
	Eref e2 = i2.eref();
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert ( syn != 0 );
	assert ( syn->getName() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( syn, di );

	/*
	/// This old utility function is replaced with the series of funcs 
	bool ret = SparseMsg::add( e2.element(), "spike", syn, "addSpike", 
		connectionProbability );
	*/
	SparseMsg* sm = new SparseMsg( e2.element(), syn );
	assert( sm );
	const Finfo* f1 = ic->findFinfo( "spike" );
	const Finfo* f2 = Synapse::initCinfo()->findFinfo( "addSpike" );
	assert( f1 && f2 );
	f1->addMsg( f2, sm->mid(), t2 );
	sm->randomConnect( connectionProbability );
	//sm->loadBalance( 1 );

	unsigned int nd = syn->dataHandler()->numData();
//	cout << "Num Syn = " << nd << endl;
	assert( nd == NUMSYN );
	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	bool ret = Field< double >::setVec( e2, "Vm", temp );
	assert( ret );
	/*
	for ( unsigned int i = 0; i < 40; ++i )
		cout << reinterpret_cast< IntFire* >( e2.element()->data( i ) )->getVm() << "	" << temp[i] << endl;
		*/
	temp.clear();
	temp.resize( size, thresh );
	ret = Field< double >::setVec( e2, "thresh", temp );
	assert( ret );
	temp.clear();
	temp.resize( size, refractoryPeriod );
	ret = Field< double >::setVec( e2, "refractoryPeriod", temp );
	assert( ret );

	vector< double > weight;
	weight.reserve( nd );
	vector< double > delay;
	delay.reserve( nd );
	for ( unsigned int i = 0; i < size; ++i ) {
		unsigned int numSyn = syne.element()->dataHandler()->numData2( i );
		for ( unsigned int j = 0; j < numSyn; ++j ) {
			weight.push_back( mtrand() * weightMax );
			delay.push_back( mtrand() * delayMax );
		}
	}
	ret = Field< double >::setVec( syne, "weight", weight );
	assert( ret );
	ret = Field< double >::setVec( syne, "delay", delay );
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
		unsigned int numWorkerThreads = 1;
		unsigned int startThread = 1;
		if ( Qinfo::numSimGroup() >= 2 ) {
			numWorkerThreads = Qinfo::simGroup( 1 )->numThreads;
			startThread = Qinfo::simGroup( 1 )->startThread;
		}
		unsigned int totOutqEntries = 0;
		for ( unsigned int j = 0; j < numWorkerThreads; ++j )
			totOutqEntries += Qinfo::outQ_[ j ].size();
		assert( totOutqEntries / ( sizeof( Qinfo ) + sizeof( double ) ) == qSize[i] );
		// cout << i << ": " << totOutqEntries / ( sizeof( Qinfo ) + sizeof( double ) ) << endl << endl ;
		// cout << p.currTime << "	" << ifire100->getVm() << "	" << ifire900->getVm() << endl;
		// cout << "T = " << p.currTime << ", Q size = " << Qinfo::q_[0].size() << endl;
		Qinfo::clearQ( &p );
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
	static const double EPSILON = 1e-9;
	const Cinfo* cc = Clock::initCinfo();
	// const Cinfo* tc = Tick::initCinfo();
	unsigned int size = 10;
	vector< unsigned int > dims( 1, 1 );
	Id clock = Id::nextId();
	// bool ret = cc->create( clock, "clock", 1 );
	Element* temp = new Element( clock, cc, "clock", dims, 1 );
	assert( temp );

	Eref clocker = clock.eref();
	Id tickId( clock.value() + 1 );
	Element* ticke = tickId();
	assert ( ticke != 0 );
	assert ( ticke->getName() == "tick" );

	assert( ticke->dataHandler()->numData() == 0 );
	bool ret = Field< unsigned int >::set( clocker, "numTicks", size );
	assert( ret );
	assert( ticke->dataHandler()->numData() == size );


	for ( unsigned int i = 0; i < size; ++i ) {
		DataId di( 0, i ); // DataId( data, field )
		Eref te( ticke, di );
		double dt = i;
		ret = Field< double >::set( te, "dt", dt );
		assert( ret );
		double val = Field< double >::get( te, "localdt" );
		assert( fabs( dt - val ) < EPSILON );

		dt *= 10.0;
		ret = Field< double >::set( te, "localdt", dt );
		assert( ret );
		val = Field< double >::get( te, "dt" );
		assert( fabs( dt - val ) < EPSILON );
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
			ProcInfo p;
			s_ = s + s_;
			s0.send( e, &p );
		}

		void handleS2( const Eref& e, const Qinfo* q, int i1, int i2 ) {
			ProcInfo p;
			i1_ += 10 * i1;
			i2_ += 10 * i2;
			s0.send( e, &p );
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
	vector< unsigned int > dims( 1, 1 );
	Element* temp = new Element( t1, Test::initCinfo(), "test1", dims, 1 );
	assert( temp );
	temp = new Element( t2, Test::initCinfo(), "test2", dims, 1 );
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
	Msg* m = new OneToOneMsg( t1(), t2() );
	assert( m != 0 );
	bool ret = shareFinfo->addMsg( shareFinfo, m->mid(), t1() );
	assert( ret );

	// Display stuff. Need to figure out how to unit test this.
	// t1()->showMsg();
	// t2()->showMsg();

	// Send messages
	ProcInfo p;
	string arg1 = " hello ";
	s1.send( t1.eref(), &p, arg1 );
	s2.send( t1.eref(), &p, 100, 200 );

	Qinfo::clearQ( &p );
	Qinfo::clearQ( &p );

	string arg2 = " goodbye ";
	s1.send( t2.eref(), &p, arg2 );
	s2.send( t2.eref(), &p, 500, 600 );

	Qinfo::clearQ( &p );
	Qinfo::clearQ( &p );

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
	
	char buf[500];

	Conv< vector< unsigned int > > intConv( intVec );
	assert( intConv.size() == sizeof( unsigned int ) * (intVec.size() + 1));
	unsigned int ret = intConv.val2buf( buf );
	assert( ret == sizeof( unsigned int ) * (intVec.size() + 1));
	assert( *reinterpret_cast< unsigned int* >( buf ) == intVec.size() );

	Conv< vector< unsigned int > > testIntConv( buf );
	assert( intConv.size() == testIntConv.size() );
	vector< unsigned int > testIntVec = *testIntConv;
	assert( intVec.size() == testIntVec.size() );
	for ( unsigned int i = 0; i < intVec.size(); ++i ) {
		assert( intVec[ i ] == testIntVec[i] );
	}

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
	vector< unsigned int > dims( 1, size );

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new Element( i1, ac, "test1", dims, 1 );
	assert( ret );
	ret = new Element( i2, ac, "test2", dims, 1 );
	assert( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( Eref( i1(), 5 ), Eref( i2(), 3 ) );
	ProcInfo p;

	Id msgElmId = m->id();

	Element *msgElm = msgElmId();

	assert( msgElm->getName() == "singleMsg" );

	Eref msgEr = m->manager( msgElmId );

	MsgManager* mm = reinterpret_cast< MsgManager* >( msgEr.data() );
	assert( mm );
	assert ( mm->getMid() == m->mid() );
	SingleMsgWrapper *sm = reinterpret_cast< SingleMsgWrapper* >( mm );
	assert( sm->getI1() == DataId( 5 ) );
	assert( sm->getI2() == DataId( 3 ) );
	
	SrcFinfo1<double> s( "test", "" );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 42;
		s.send( Eref( e1.element(), i ), &p, x );
	}
	Qinfo::clearQ( &p );

	// Check that regular msgs go through.
	Eref tgt3( i2(), 3 );
	Eref tgt8( i2(), 8 );
	double val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	assert( fabs( val - 5 * 42 ) < 1e-8 );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	assert( fabs( val ) < 1e-8 );

	// Now change I1 and I2, rerun, and check.
	sm->setI1( 9 );
	sm->setI2( 8 );
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 1000;
		s.send( Eref( e1.element(), i ), &p, x );
	}
	Qinfo::clearQ( &p );
	val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	assert( fabs( val - 5 * 42 ) < 1e-8 );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	assert( fabs( val - 9000 ) < 1e-8 );

	cout << "." << flush;

	delete i1();
	delete i2();
}

void testSetGetExtField()
{
	static const double EPSILON = 1e-9;
	const Cinfo* nc = Neutral::initCinfo();
	const Cinfo* rc = Mdouble::initCinfo();
	unsigned int size = 100;
	vector< unsigned int > dims( 1, size );
	string arg;
	Id i1 = Id::nextId();
	Id i2( i1.value() + 1 );
	Id i3( i2.value() + 1 );
	Id i4( i3.value() + 1 );
	Element* e1 = new Element( i1, nc, "test", dims, 1 );
	assert( e1 );
	Shell::adopt( Id(), i1 );
	Element* e2 = new Element( i2, rc, "x", dims, 1 );
	assert( e2 );
	Shell::adopt( i1, i2 );
	Element* e3 = new Element( i3, rc, "y", dims, 1 );
	assert( e3 );
	Shell::adopt( i1, i3 );
	Element* e4 = new Element( i4, rc, "z", dims, 1 );
	assert( e4 );
	Shell::adopt( i1, i4 );
	bool ret;

	vector< double > vec;
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref a( e1, i );
		Eref b( e1, size - i - 1 );
		double temp = i;
		ret = Field< double >::set( a, "x", temp );
		assert( ret );
		double temp2  = temp * temp;
		ret = Field< double >::set( b, "y", temp2 );
		assert( ret );
		vec.push_back( temp2 - temp );
	}

	ret = Field< double >::setVec( Eref( e1, 0 ), "z", vec );
	assert( ret );

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref a( e2, i );
		Eref b( e3, size - i - 1 );
		Eref c( e4, i );
		double temp = i;
		double temp2  = temp * temp;

		double v = reinterpret_cast< Mdouble* >(a.data() )->getThis();
		assert( fabs ( v - temp ) < EPSILON ); 

		v = reinterpret_cast< Mdouble* >(b.data() )->getThis();
		assert( fabs( v - temp2 ) < EPSILON );

		v = reinterpret_cast< Mdouble* >( c.data() )->getThis();
		assert( fabs( v - ( temp2 - temp ) ) < EPSILON );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref a( e1, i );
		Eref b( e1, size - i - 1 );

		double temp = i;
		double temp2  = temp * temp;
		double ret = Field< double >::get( a, "x" );
		assert( fabs ( temp - ret ) < EPSILON );
		
		ret = Field< double >::get( b, "y" );
		assert( fabs ( temp2 - ret ) < EPSILON );

		ret = Field< double >::get( a, "z" );
		assert( fabs ( (temp2 - temp) - ret ) < EPSILON );
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

void testIsA()
{
	static const double EPSILON = 1e-9;
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

void testAsync( )
{
	showFields();
	testPrepackedBuffer();
	insertIntoQ();
	testSendMsg();
	testCreateMsg();
	testSet();
	testGet();
	testSetGet();
	testSetGetDouble();
	testSetGetSynapse();
	testSetGetVec();
	testSetRepeat();
	testStrSet();
	testStrGet();
	testSendSpike();
	testSparseMatrix();
	testSparseMatrix2();
	testSparseMatrixBalance();
	testSparseMsg();
	testUpValue();
	testSharedMsg();
	testConvVector();
	testMsgField();
	testSetGetExtField();
	testIsA();
}
