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
#include "Dinfo.h"
#include <queue>
#include "../biophysics/SpikeRingBuffer.h"
#include "../biophysics/Synapse.h"
#include "../biophysics/SynHandler.h"
#include "../biophysics/IntFire.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "OneToOneMsg.h"
#include "../randnum/randnum.h"
#include "../scheduling/Clock.h"

#include "../shell/Shell.h"
#include "../mpi/PostMaster.h"

void showFields()
{
	const Cinfo* nc = Neutral::initCinfo();
	Id i1 = Id::nextId();
	Element* ret = new GlobalDataElement( i1, nc, "test1", 1 );
	MOOSE_ASSERT( ret );
	// i1.eref().element()->showFields();
	cout << "." << flush;

	delete i1.element();
}

void testSendMsg()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	const DestFinfo* df = dynamic_cast< const DestFinfo* >(
		ac->findFinfo( "setOutputValue" ) );
	MOOSE_ASSERT( df != 0 );
	FuncId fid = df->getFid();

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new GlobalDataElement( i1, ac, "test1", size );
	// bool ret = nc->create( i1, "test1", size );
	MOOSE_ASSERT( ret );
	// ret = nc->create( i2, "test2", size );
	ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new OneToOneMsg( e1, e2, 0 );
	vector< vector< Eref > > ver;
	m->targets( ver );
	MOOSE_ASSERT( ver.size() == size );
	MOOSE_ASSERT( ver[0].size() == 1 );
	MOOSE_ASSERT( ver[0][0].element() == e2.element() );
	MOOSE_ASSERT( ver[0][0].dataIndex() == e2.dataIndex() );
	MOOSE_ASSERT( ver[55].size() == 1 );
	MOOSE_ASSERT( ver[55][0].element() == e2.element() );
	MOOSE_ASSERT( ver[55][0].dataIndex() == 55 );
	
	SrcFinfo1<double> s( "test", "" );
	s.setBindIndex( 0 );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );
	// e1.element()->digestMessages();
	const vector< MsgDigest >& md = e1.element()->msgDigest( 0 );
	MOOSE_ASSERT( md.size() == 1 );
	MOOSE_ASSERT( md[0].targets.size() == 1 );
	MOOSE_ASSERT( md[0].targets[0].element() == e2.element() );
	MOOSE_ASSERT( md[0].targets[0].dataIndex() == e2.dataIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i + i * i;
		s.send( Eref( e1.element(), i ), x );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i + i * i;
		double val = reinterpret_cast< Arith* >(e2.element()->data( i ) )->getOutput();
		MOOSE_ASSERT( doubleEq( val, temp ) );
	}
	cout << "." << flush;

	delete i1.element();
	delete i2.element();
}

// This used to use parent/child msg, but that has other implications
// as it causes deletion of elements.
void testCreateMsg()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* temp = new GlobalDataElement( i1, ac, "test1", size );
	// bool ret = nc->create( i1, "test1", size );
	MOOSE_ASSERT( temp );
	temp = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( temp );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	OneToOneMsg *m = new OneToOneMsg( e1, e2, 0 );
	MOOSE_ASSERT( m );
	const Finfo* f1 = ac->findFinfo( "output" );
	MOOSE_ASSERT( f1 );
	const Finfo* f2 = ac->findFinfo( "arg1" );
	MOOSE_ASSERT( f2 );
	bool ret = f1->addMsg( f2, m->mid(), e1.element() );
	
	MOOSE_ASSERT( ret );
	// e1.element()->digestMessages();

	for ( unsigned int i = 0; i < size; ++i ) {
		const SrcFinfo1< double >* sf = dynamic_cast< const SrcFinfo1< double >* >( f1 );
		MOOSE_ASSERT( sf != 0 );
		sf->send( Eref( e1.element(), i ), double( i ) );
		double val = reinterpret_cast< Arith* >(e2.element()->data( i ) )->getArg1();
		MOOSE_ASSERT( doubleEq( val, i ) );
	}

	/*
	for ( unsigned int i = 0; i < size; ++i )
		cout << i << "	" << reinterpret_cast< Neutral* >(e2.element()->data( i ))->getName() << endl;

*/
	cout << "." << flush;
	delete i1.element();
	delete i2.element();
}

void testSetGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Element* ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );
	ProcInfo p;
	
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double x = i * 3.14;
		bool ret = Field< double >::set( oid, "outputValue", x );
		MOOSE_ASSERT( ret );
		double val = reinterpret_cast< Arith* >(oid.data())->getOutput();
		MOOSE_ASSERT( doubleEq( val, x ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double x = i * 3.14;
		double ret = Field< double >::get( oid, "outputValue" );
		ProcInfo p;
		MOOSE_ASSERT( doubleEq( ret, x ) );
	}

	cout << "." << flush;
	delete i2.element();
}

void testStrSet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Element* ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );
	ProcInfo p;

	Shell::adopt( Id(), i2, 0 );

	MOOSE_ASSERT( ret->getName() == "test2" );
	bool ok = SetGet::strSet( ObjId( i2, 0 ), "name", "NewImprovedTest" );
	MOOSE_ASSERT( ok );
	MOOSE_ASSERT( ret->getName() == "NewImprovedTest" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = sqrt((double) i );
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );
		stringstream ss;
		ss << setw( 10 ) << x;
		ok = SetGet::strSet( dest, "outputValue", ss.str() );
		MOOSE_ASSERT( ok );
		// SetGet1< double >::set( dest, "setOutputValue", x );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = sqrt((double) i );
		double val = reinterpret_cast< Arith* >( 
						Eref( i2.element(), i ).data() )->getOutput();
		MOOSE_ASSERT( fabs( val - temp ) < 1e-5 );
		// DoubleEq won't work here because string is truncated.
	}

	cout << "." << flush;

	delete i2.element();
}

void testGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	Element* ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );
	ProcInfo p;

	ObjId oid( i2, 0 );

	string val = Field< string >::get( oid, "name" );
	MOOSE_ASSERT( val == "test2" );
	ret->setName( "HupTwoThree" );
	val = Field< string >::get( oid, "name" );
	MOOSE_ASSERT( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >(oid.element()->data( i ))->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );

		double val = Field< double >::get( dest, "outputValue" );
		double temp = i * 3;
		MOOSE_ASSERT( doubleEq( val, temp ) );
	}

	cout << "." << flush;
	delete i2.element();
}

void testStrGet()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = Id::nextId();

	Element* ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );
	ProcInfo p;

	ObjId oid( i2, 0 );

	string val;
	bool ok = SetGet::strGet( oid, "name", val );
	MOOSE_ASSERT( ok );
	MOOSE_ASSERT( val == "test2" );
	ret->setName( "HupTwoThree" );
	ok = SetGet::strGet( oid, "name", val );
	MOOSE_ASSERT( ok );
	MOOSE_ASSERT( val == "HupTwoThree" );
	
	for ( unsigned int i = 0; i < size; ++i ) {
		double temp = i * 3;
		reinterpret_cast< Arith* >( ObjId( i2, i ).data() )->setOutput( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref dest( e2.element(), i );
		ObjId dest( i2, i );
		ok = SetGet::strGet( dest, "outputValue", val );
		MOOSE_ASSERT( ok );
		double conv = atof( val.c_str() );
		double temp = i * 3;
		MOOSE_ASSERT( fabs( conv - temp ) < 1e-5 );
		// DoubleEq won't work here because string is truncated.
	}

	cout << "." << flush;
	delete i2.element();
}


void testSetGetDouble()
{
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Id i3( i2.value() + 1 );
	Element* ret = new GlobalDataElement( i2, ic, "test2", size );
	MOOSE_ASSERT( ret );
	ProcInfo p;

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref e2( i2(), i );
		ObjId oid( i2, i );
		double temp = i;
		bool ret = Field< double >::set( oid, "Vm", temp );
		MOOSE_ASSERT( ret );
		MOOSE_ASSERT( 
			doubleEq ( reinterpret_cast< IntFire* >(oid.data())->getVm() , temp ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId oid( i2, i );
		double temp = i;
		double ret = Field< double >::get( oid, "Vm" );
		MOOSE_ASSERT( doubleEq( temp, ret ) );
	}

	cout << "." << flush;
	delete i2.element();
	delete i3.element();
}

void testSetGetSynapse()
{
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 100;

	string arg;
	Id cells = Id::nextId();
	Element* temp = new GlobalDataElement( cells, ic, "test2", size );
	MOOSE_ASSERT( temp );
	vector< unsigned int > ns( size );
	vector< vector< double > > delay( size );
	for ( unsigned int i = 0; i < size; ++i ) {
		ns[i] = i;
		for ( unsigned int j = 0; j < i; ++j ) {
			double temp = i * 1000 + j;
			delay[i].push_back( temp );
		}
	}

	bool ret = Field< unsigned int >::setVec( cells, "numSynapse", ns );
	MOOSE_ASSERT( ret );
	MOOSE_ASSERT( temp->numData() == size );
	Id syns( cells.value() + 1 );
	for ( unsigned int i = 0; i < size; ++i ) {
		ret = Field< double >::
				setVec( ObjId( syns, i ), "delay", delay[i] );
		if ( i > 0 )
			MOOSE_ASSERT( ret );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		MOOSE_ASSERT( syns.element()->numField( i ) == i );
		IntFire* fire = reinterpret_cast< IntFire* >( temp->data( i ) );
		MOOSE_ASSERT( fire->getNumSynapses() == i );
		for ( unsigned int j = 0; j < i; ++j ) {
			// ObjId oid( syns, i, j );
			ObjId oid( syns, i, j );
			double x = i * 1000 + j ;
			double d = Field< double >::get( oid, "delay" );
			double d2 = fire->getSynapse( j )->getDelay();
			MOOSE_ASSERT( doubleEq( d, x ) );
			MOOSE_ASSERT( doubleEq( d2, x ) );
		}
	}
	delete syns.element();
	delete temp;
	cout << "." << flush;
}

void testSetGetVec()
{
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Element* temp = new GlobalDataElement( i2, ic, "test2", size );
	MOOSE_ASSERT( temp );

	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2.element(), 0 );
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( i2, "numSynapse", numSyn );
	MOOSE_ASSERT( ret );

	for ( unsigned int i = 0; i < size; ++i ) {
		IntFire* fire = reinterpret_cast< IntFire* >( i2.element()->data( i ) );
		MOOSE_ASSERT( fire->getNumSynapses() == i );
	}

	vector< unsigned int > getSyn;

	Field< unsigned int >::getVec( i2, "numSynapse", getSyn );
	MOOSE_ASSERT (getSyn.size() == size );
	for ( unsigned int i = 0; i < size; ++i )
		MOOSE_ASSERT( getSyn[i] == i );

	Id synapse( i2.value() + 1 );
	delete synapse.element();
	delete temp;
	cout << "." << flush;
}

void testSendSpike()
{
	static const double WEIGHT = -1.0;
	static const double TAU = 1.0;
	static const double DT = 0.1;
	const Cinfo* ic = IntFire::initCinfo();
	const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Element* temp = new GlobalDataElement( i2, ic, "test2", size );
	MOOSE_ASSERT( temp );
	Eref e2 = i2.eref();
	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref er( i2(), i );
		ObjId oid( i2, i );
		bool ret = Field< unsigned int >::set( oid, "numSynapses", i );
		MOOSE_ASSERT( ret );
	}

	Id synId( i2.value() + 1 );
	ObjId target( synId , 1 );

	reinterpret_cast< Synapse* >(target.data())->setWeight( WEIGHT );
	reinterpret_cast< Synapse* >(target.data())->setDelay( 0.01 );
	SingleMsg *m = new SingleMsg( e2, target.eref(), 0 );
	const Finfo* f1 = ic->findFinfo( "spikeOut" );
	const Finfo* f2 = sc->findFinfo( "addSpike" );
	bool ret = f1->addMsg( f2, m->mid(), e2.element() );
	MOOSE_ASSERT( ret );

	reinterpret_cast< IntFire* >(e2.data())->setVm( 1.0 );
	// ret = SetGet1< double >::set( e2, "Vm", 1.0 );
	ProcInfo p;
	p.dt = DT;
	reinterpret_cast< IntFire* >(e2.data())->process( e2, &p );
	// At this stage we have sent the spike, so e2.data::Vm should be -1e-7.
	double Vm = reinterpret_cast< IntFire* >(e2.data())->getVm();
	MOOSE_ASSERT( doubleEq( Vm, -1e-7 ) );
	ObjId targetCell( i2, 1 );
	reinterpret_cast< IntFire* >(targetCell.data())->setTau( TAU );

	reinterpret_cast< IntFire* >(targetCell.data())->process( targetCell.eref(), &p );
	Vm = Field< double >::get( targetCell, "Vm" );
	MOOSE_ASSERT( doubleEq( Vm , WEIGHT * ( 1.0 - DT / TAU ) ) );
	cout << "." << flush;
	delete i2.element();
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
	
	static unsigned int dropN[] = { 1, 6, 2, 7 };
	static unsigned int dropColIndex[] = { 0, 1, 0, 1 };

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
			MOOSE_ASSERT( n[j] == preN[ k ] );
			MOOSE_ASSERT( c[j] == preColIndex[ k ] );
			k++;
		}
	}
	MOOSE_ASSERT( k == 7 );

	// printSparseMatrix( m );

	m.transpose();
	MOOSE_ASSERT( m.nRows() == nCols );
	MOOSE_ASSERT( m.nColumns() == nRows );

	k = 0;
	for ( unsigned int i = 0; i < nCols; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j ) {
			MOOSE_ASSERT( n[j] == postN[ k ] );
			MOOSE_ASSERT( c[j] == postColIndex[ k ] );
			k++;
		}
	}
	MOOSE_ASSERT( k == 7 );

	// Drop column 1.
	vector< unsigned int > keepCols( 2 );
	keepCols[0] = 0;
	keepCols[1] = 2;
	// cout << endl; m.print();
	m.reorderColumns( keepCols );
	// cout << endl; m.print();
	MOOSE_ASSERT( m.nRows() == nCols );
	MOOSE_ASSERT( m.nColumns() == 2 );

	k = 0;
	for ( unsigned int i = 0; i < nCols; ++i ) {
		unsigned int num = m.getRow( i, &n, &c );
		for ( unsigned int j = 0; j < num; ++j ) {
			MOOSE_ASSERT( n[j] == dropN[ k ] );
			MOOSE_ASSERT( c[j] == dropColIndex[ k ] );
			k++;
		}
	}
	MOOSE_ASSERT( k == 4 );

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
			MOOSE_ASSERT (n.get( j, i ) ==  m[i][j] );
	n.transpose();
	for ( unsigned int i = 0; i < 10; ++i )
		for ( unsigned int j = 0; j < 10; ++j )
			MOOSE_ASSERT (n.get( i, j ) ==  m[i][j] );

	///////////////////////////////////////////////////////////////
	// Drop columns 2 and 7.
	///////////////////////////////////////////////////////////////
	static unsigned int init[] = {0, 1, 3, 4, 5, 6, 8, 9};
	vector< unsigned int > keepCols( 
					init, init + sizeof( init ) / sizeof( unsigned int ) );
	n.reorderColumns( keepCols );
	for ( unsigned int i = 0; i < 10; ++i ) {
		for ( unsigned int j = 0; j < 8; ++j ) {
			unsigned int k = keepCols[j];
			MOOSE_ASSERT (n.get( i, j ) ==  m[i][k] );
		}
	}
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

void testSparseMatrixReorder()
{
	SparseMatrix< int > n( 2, 1 );
	n.set( 0, 0, -1 );
	n.set( 1, 0, 1 );
	vector< unsigned int > colOrder( 1, 0 ); // Keep the original as is
	n.reorderColumns( colOrder ); // This case failed in an earlier version
	MOOSE_ASSERT( n.get( 0, 0 ) == -1 );
	MOOSE_ASSERT( n.get( 1, 0 ) == 1 );

	unsigned int nrows = 4;
	unsigned int ncolumns = 5;

	//////////////////////////////////////////////////////////////
	// Test a reordering
	//////////////////////////////////////////////////////////////
	n.setSize( nrows, ncolumns );
	for ( unsigned int i = 0; i < nrows; ++i ) {
		for ( unsigned int j = 0; j < ncolumns; ++j ) {
			int x = i * 10 + j;
			n.set( i, j, x );
		}
	}
	colOrder.resize( ncolumns );
	colOrder[0] = 3;
	colOrder[1] = 2;
	colOrder[2] = 0;
	colOrder[3] = 4;
	colOrder[4] = 1;
	n.reorderColumns( colOrder );
	MOOSE_ASSERT( n.nRows() == nrows );
	MOOSE_ASSERT( n.nColumns() == ncolumns );
	for ( unsigned int i = 0; i < nrows; ++i ) {
		for ( unsigned int j = 0; j < ncolumns; ++j ) {
			int x = i * 10 + colOrder[j];
			MOOSE_ASSERT( n.get( i, j ) == x );
		}
	}

	//////////////////////////////////////////////////////////////
	// Test reordering + eliminating some columns
	//////////////////////////////////////////////////////////////
	// Put back in original config
	for ( unsigned int i = 0; i < nrows; ++i ) {
		for ( unsigned int j = 0; j < ncolumns; ++j ) {
			unsigned int x = i * 10 + j;
			n.set( i, j, x );
		}
	}
	colOrder.resize( 2 );
	colOrder[0] = 3;
	colOrder[1] = 2;
	n.reorderColumns( colOrder );
	MOOSE_ASSERT( n.nRows() == nrows );
	MOOSE_ASSERT( n.nColumns() == 2 );
	for ( unsigned int i = 0; i < nrows; ++i ) {
		MOOSE_ASSERT( n.get( i, 0 ) == static_cast< int >( i * 10 + 3 ) );
		MOOSE_ASSERT( n.get( i, 1 ) == static_cast< int >( i * 10 + 2 ) );
	}
	cout << "." << flush;
}

void testSparseMatrixFill()
{
	SparseMatrix< int > n;
	unsigned int nrow = 5;
	unsigned int ncol = 7;
	vector< unsigned int > row;
	vector< unsigned int > col;
	vector< int > val;
	unsigned int num = 0;
	for ( unsigned int i = 0; i < nrow; ++i ) {
		for ( unsigned int j = 0; j < ncol; ++j ) {
			if ( j == 0 || i + j == 6 || ( j - i) == 2 ) {
				row.push_back( i );
				col.push_back( j );
				val.push_back( 100 + i * 10 + j );
				++num;
			}
		}
	}
	n.tripletFill( row, col, val );
	// n.print();
	MOOSE_ASSERT( n.nRows() == nrow );
	MOOSE_ASSERT( n.nColumns() == ncol );
	MOOSE_ASSERT( n.nEntries() == num );
	for ( unsigned int i = 0; i < nrow; ++i ) {
		for ( unsigned int j = 0; j < ncol; ++j ) {
			int val = n.get( i, j );
			if ( j == 0 || i + j == 6 || ( j - i) == 2 )
				MOOSE_ASSERT( static_cast< unsigned int >( val ) == 100 + i * 10 + j );
			else
				MOOSE_ASSERT( val == 0 );
		}
	}
	cout << "." << flush;
}

void printGrid( Element* e, const string& field, double min, double max )
{
	static string icon = " .oO@";
	unsigned int yside = sqrt( double ( e->numData() ) );
	unsigned int xside = e->numData() / yside;
	if ( e->numData() % yside > 0 )
		xside++;
	
	for ( unsigned int i = 0; i < e->numData(); ++i ) {
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
	// static const unsigned int NUMSYN = 104576;
	static const double thresh = 0.2;
	static const double Vmax = 1.0;
	static const double refractoryPeriod = 0.4;
	static const double weightMax = 0.02;
	static const double delayMax = 4;
	static const double timestep = 0.2;
	static const double connectionProbability = 0.1;
	static const unsigned int runsteps = 5;
	const Cinfo* ic = IntFire::initCinfo();
	const Cinfo* sc = Synapse::initCinfo();
	const Finfo* procFinfo = ic->findFinfo( "process" );
	MOOSE_ASSERT( procFinfo );
	const DestFinfo* df = dynamic_cast< const DestFinfo* >( procFinfo );
	MOOSE_ASSERT( df );
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 1024;

	string arg;

	mtseed( 5489UL ); // The default value, but better to be explicit.

	Id cells = Id::nextId();
	// bool ret = ic->create( cells, "test2", size );
	Element* t2 = new GlobalDataElement( cells, ic, "test2", size );
	MOOSE_ASSERT( t2 );
	Id syns( cells.value() + 1 );

	SparseMsg* sm = new SparseMsg( t2, syns.element(), 0 );
	MOOSE_ASSERT( sm );
	const Finfo* f1 = ic->findFinfo( "spikeOut" );
	const Finfo* f2 = sc->findFinfo( "addSpike" );
	MOOSE_ASSERT( f1 && f2 );
	f1->addMsg( f2, sm->mid(), t2 );
	sm->randomConnect( connectionProbability );

	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	bool ret = Field< double >::setVec( cells, "Vm", temp );
	MOOSE_ASSERT( ret );
	temp.clear();
	temp.resize( size, thresh );
	ret = Field< double >::setVec( cells, "thresh", temp );
	MOOSE_ASSERT( ret );
	temp.clear();
	temp.resize( size, refractoryPeriod );
	ret = Field< double >::setVec( cells, "refractoryPeriod", temp );
	MOOSE_ASSERT( ret );

	unsigned int fieldSize = 5000;
	vector< double > weight( size * fieldSize, 0.0 );
	vector< double > delay( size * fieldSize, 0.0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId id( cells, i );
		unsigned int numSyn = 
				Field< unsigned int >::get( id, "numSynapse" );
		unsigned int k = i * fieldSize;
		for ( unsigned int j = 0; j < numSyn; ++j ) {
			weight[ k + j ] = mtrand() * weightMax;
			delay[ k + j ] = mtrand() * delayMax;
		}
	}
	ret = Field< double >::setVec( syns, "weight", weight );
	MOOSE_ASSERT( ret );
	ret = Field< double >::setVec( syns, "delay", delay );
	MOOSE_ASSERT( ret );

	// printGrid( cells(), "Vm", 0, thresh );

	ProcInfo p;
	p.dt = timestep;
	for ( unsigned int i = 0; i < runsteps; ++i ) {
		p.currTime += p.dt;
		SetGet1< ProcInfo* >::setRepeat( cells, "process", &p );
		// cells()->process( &p, df->getFid() );
	}

	delete syns.element();
	delete cells.element();
	cout << "." << flush;
}

void test2ArgSetVec()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i2 = Id::nextId();
	Element* ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );

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
		MOOSE_ASSERT( doubleEq( val, x ) );
	}
	cout << "." << flush;
	delete i2.element();
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

	string arg;
	Id cell = Id::nextId();
	// bool ret = ic->create( i2, "test2", size );
	Element* temp = new GlobalDataElement( cell, ic, "cell", size );
	MOOSE_ASSERT( temp );
	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	// Here we test setting a 1-D vector
	bool ret = Field< unsigned int >::setVec( cell, "numSynapse", numSyn);
	MOOSE_ASSERT( ret );
	
	Id synapse( cell.value() + 1 );
	// Here we test setting a 2-D array with different dims on each axis.
	for ( unsigned int i = 0; i < size; ++i ) {
		ret = Field< double >::
				setRepeat( ObjId( synapse, i ), "delay", 123.0 );
		MOOSE_ASSERT( ret );
	}
	for ( unsigned int i = 0; i < size; ++i ) {
		vector< double > delay;
		Field< double >::getVec( ObjId( synapse, i ), "delay", delay );
		MOOSE_ASSERT( delay.size() == i );
		for ( unsigned int j = 0; j < i; ++j ) {
			MOOSE_ASSERT( doubleEq( delay[j], 123.0 ) );
		}
	}

	delete synapse.element();
	delete temp;
	cout << "." << flush;
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

		void handleS1( const Eref& e, string s ) {
			s_ = s + s_;
			s0.send( e );
		}

		void handleS2( const Eref& e, int i1, int i2 ) {
			i1_ += 10 * i1;
			i2_ += 10 * i2;
			s0.send( e );
		}

		static Finfo* sharedVec[ 6 ];

		static const Cinfo* initCinfo()
		{
			static SharedFinfo shared( "shared", "",
				sharedVec, sizeof( sharedVec ) / sizeof( const Finfo * ) );
			static Finfo * testFinfos[] = {
				&shared,
			};

			static Dinfo< Test > dinfo;
			static Cinfo testCinfo(
				"Test",
				0,
				testFinfos,
				sizeof( testFinfos ) / sizeof( Finfo* ),
				&dinfo
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

	Element* temp = new GlobalDataElement( t1, Test::initCinfo(), "test1", 1 );
	MOOSE_ASSERT( temp );
	temp = new GlobalDataElement( t2, Test::initCinfo(), "test2", 1 );
	// ret = Test::initCinfo()->create( t2, "test2", 1 );
	MOOSE_ASSERT( temp );

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
	MOOSE_ASSERT( shareFinfo != 0 );
	Msg* m = new OneToOneMsg( t1.eref(), t2.eref(), 0 );
	MOOSE_ASSERT( m != 0 );
	bool ret = shareFinfo->addMsg( shareFinfo, m->mid(), t1.element() );
	MOOSE_ASSERT( ret );

	// t1.element()->digestMessages();
	// t2.element()->digestMessages();
	// Display stuff. Need to figure out how to unit test this.
	// t1()->showMsg();
	// t2()->showMsg();

	// Send messages
	ProcInfo p;
	string arg1 = " hello ";
	s1.send( t1.eref(), arg1 );
	s2.send( t1.eref(), 100, 200 );

	string arg2 = " goodbye ";
	s1.send( t2.eref(), arg2 );
	s2.send( t2.eref(), 500, 600 );

	/*
	cout << "data1: s=" << tdata1->s_ << 
		", i1=" << tdata1->i1_ << ", i2=" << tdata1->i2_ << 
		", numAcks=" << tdata1->numAcks_ << endl;
	cout << "data2: s=" << tdata2->s_ << 
		", i1=" << tdata2->i1_ << ", i2=" << tdata2->i2_ <<
		", numAcks=" << tdata2->numAcks_ << endl;
	*/
	// Check results
	
	MOOSE_ASSERT( tdata1->s_ == " goodbye tdata1" );
	MOOSE_ASSERT( tdata2->s_ == " hello TDATA2" );
	MOOSE_ASSERT( tdata1->i1_ == 5001  );
	MOOSE_ASSERT( tdata1->i2_ == 6002  );
	MOOSE_ASSERT( tdata2->i1_ == 1005  );
	MOOSE_ASSERT( tdata2->i2_ == 2006  );
	MOOSE_ASSERT( tdata1->numAcks_ == 2  );
	MOOSE_ASSERT( tdata2->numAcks_ == 2  );
	
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
	double* tempBuf = buf;

	Conv< vector< unsigned int > > intConv;
	MOOSE_ASSERT( intConv.size( intVec ) == 1 + intVec.size() );
	intConv.val2buf( intVec, &tempBuf );
	MOOSE_ASSERT( tempBuf == buf + 6 );
	MOOSE_ASSERT( buf[0] == intVec.size() );
	MOOSE_ASSERT( static_cast< unsigned int >( buf[1] ) == intVec[0] );
	MOOSE_ASSERT( static_cast< unsigned int >( buf[2] ) == intVec[1] );
	MOOSE_ASSERT( static_cast< unsigned int >( buf[3] ) == intVec[2] );
	MOOSE_ASSERT( static_cast< unsigned int >( buf[4] ) == intVec[3] );
	MOOSE_ASSERT( static_cast< unsigned int >( buf[5] ) == intVec[4] );

	tempBuf = buf;
	const vector< unsigned int >& testIntVec = intConv.buf2val( &tempBuf );

	MOOSE_ASSERT( intVec.size() == testIntVec.size() );
	for ( unsigned int i = 0; i < intVec.size(); ++i ) {
		MOOSE_ASSERT( intVec[ i ] == testIntVec[i] );
	}

	vector< string > strVec;
	strVec.push_back( "one" );
	strVec.push_back( "two" );
	strVec.push_back( "three and more and more and more" );
	strVec.push_back( "four and yet more" );

	tempBuf = buf;
	Conv< vector< string > >::val2buf( strVec, &tempBuf );
	unsigned int sz = Conv< vector< string > >::size( strVec );
	MOOSE_ASSERT( sz == 1 + 2 + ( strVec[2].length() + 8) /8 + ( strVec[3].length() + 8 )/8 );
	MOOSE_ASSERT( buf[0] == 4 );
	MOOSE_ASSERT( strcmp( reinterpret_cast< char* >( buf + 1 ), "one" ) == 0 );
	
	tempBuf = buf;
	const vector< string >& tgtStr = 
			Conv< vector< string > >::buf2val( &tempBuf );
	MOOSE_ASSERT( tgtStr.size() == 4 );
	for ( unsigned int i = 0; i < 4; ++i )
		MOOSE_ASSERT( tgtStr[i] == strVec[i] );

	cout << "." << flush;
}

void testConvVectorOfVectors()
{
	short *row0 = 0;
	short row1[] = { 1 };
	short row2[] = { 2, 3 };
	short row3[] = { 4, 5, 6 };
	short row4[] = { 7, 8, 9, 10 };
	short row5[] = { 11, 12, 13, 14, 15 };

	vector< vector < short > > vec( 6 );
	vec[0].insert( vec[0].end(), row0, row0 + 0 );
	vec[1].insert( vec[1].end(), row1, row1 + 1 );
	vec[2].insert( vec[2].end(), row2, row2 + 2 );
	vec[3].insert( vec[3].end(), row3, row3 + 3 );
	vec[4].insert( vec[4].end(), row4, row4 + 4 );
	vec[5].insert( vec[5].end(), row5, row5 + 5 );

	double expected[] = { 
		6,  // Number of sub-vectors
   		0,		// No entries on first sub-vec
		1,		1,
		2,		2,3,
		3,		4,5,6,
		4,		7,8,9,10,
		5,		11,12,13,14,15
	};

	double origBuf[500];
	double* buf = origBuf;

	Conv< vector< vector< short > > > conv;

	MOOSE_ASSERT( conv.size( vec ) == 1 + 6 + 0 + 1 + 2 + 3 + 4 + 5 ); // 21
	conv.val2buf( vec, &buf );
	MOOSE_ASSERT( buf == 22 + origBuf );
	for ( unsigned int i = 0; i < 22; ++i )
		MOOSE_ASSERT( doubleEq( origBuf[i], expected[i] ) );
	
	double* buf2 = origBuf;
	const vector< vector< short > >& rc = conv.buf2val( &buf2 );
	
	MOOSE_ASSERT( rc.size() == 6 );
	for ( unsigned int i = 0; i < 6; ++i ) {
		MOOSE_ASSERT( rc[i].size() == i );
		for ( unsigned int j = 0; j < i; ++j )
			MOOSE_ASSERT( rc[i][j] == vec[i][j] );
	}

	cout << "." << flush;
}

void testMsgField()
{
	const Cinfo* ac = Arith::initCinfo();
	unsigned int size = 10;

	const DestFinfo* df = dynamic_cast< const DestFinfo* >(
		ac->findFinfo( "setOutputValue" ) );
	MOOSE_ASSERT( df != 0 );
	FuncId fid = df->getFid();

	Id i1 = Id::nextId();
	Id i2 = Id::nextId();
	Element* ret = new GlobalDataElement( i1, ac, "test1", size );
	MOOSE_ASSERT( ret );
	ret = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( ret );

	Eref e1 = i1.eref();

	Msg* m = new SingleMsg( Eref( i1.element(), 5 ), Eref( i2.element(), 3 ), 0 );
	ProcInfo p;

	MOOSE_ASSERT( m->mid().element()->getName() == "singleMsg" );

	SingleMsg* sm = reinterpret_cast< SingleMsg* >( m->mid().data() );
	MOOSE_ASSERT( sm );
	MOOSE_ASSERT ( sm == m );
	MOOSE_ASSERT( sm->getI1() == 5 );
	MOOSE_ASSERT( sm->getI2() == 3 );
	
	SrcFinfo1<double> s( "test", "" );
	s.setBindIndex( 0 );
	e1.element()->addMsgAndFunc( m->mid(), fid, s.getBindIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 42;
		s.send( Eref( e1.element(), i ), x );
	}

	// Check that regular msgs go through.
	Eref tgt3( i2.element(), 3 );
	Eref tgt8( i2.element(), 8 );
	double val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	MOOSE_ASSERT( doubleEq( val, 5 * 42 ) );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	MOOSE_ASSERT( doubleEq( val, 0 ) );

	// Now change I1 and I2, rerun, and check.
	sm->setI1( 9 );
	sm->setI2( 8 );
	for ( unsigned int i = 0; i < size; ++i ) {
		double x = i * 1000;
		s.send( Eref( e1.element(), i ), x );
	}
	val = reinterpret_cast< Arith* >( tgt3.data() )->getOutput();
	MOOSE_ASSERT( doubleEq( val, 5 * 42 ) );
	val = reinterpret_cast< Arith* >( tgt8.data() )->getOutput();
	MOOSE_ASSERT( doubleEq( val, 9000 ) );

	cout << "." << flush;

	delete i1.element();
	delete i2.element();
}

void testSetGetExtField()
{
	const Cinfo* nc = Neutral::initCinfo();
	const Cinfo* rc = Arith::initCinfo();
	unsigned int size = 100;

	string arg;
	Id i1 = Id::nextId();
	Id i2( i1.value() + 1 );
	Id i3( i2.value() + 1 );
	Id i4( i3.value() + 1 );
	Element* e1 = new GlobalDataElement( i1, nc, "test", size );
	MOOSE_ASSERT( e1 );
	Shell::adopt( Id(), i1, 0 );
	Element* e2 = new GlobalDataElement( i2, rc, "x", size );
	MOOSE_ASSERT( e2 );
	Shell::adopt( i1, i2, 0 );
	Element* e3 = new GlobalDataElement( i3, rc, "y", size );
	MOOSE_ASSERT( e3 );
	Shell::adopt( i1, i3, 0 );
	Element* e4 = new GlobalDataElement( i4, rc, "z", size );
	MOOSE_ASSERT( e4 );
	Shell::adopt( i1, i4, 0 );
	bool ret;

	vector< double > vec;
	for ( unsigned int i = 0; i < size; ++i ) {
		ObjId a( i1, i );
		ObjId b( i1, size - i - 1);
		// Eref a( e1, i );
		// Eref b( e1, size - i - 1 );
		double temp = i;
		ret = Field< double >::set( a, "x", temp );
		MOOSE_ASSERT( ret );
		double temp2  = temp * temp;
		ret = Field< double >::set( b, "y", temp2 );
		MOOSE_ASSERT( ret );
		vec.push_back( temp2 - temp );
	}

	ret = Field< double >::setVec( i1, "z", vec );
	MOOSE_ASSERT( ret );

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

		double v = reinterpret_cast< Arith* >(a.data() )->getOutput();
		MOOSE_ASSERT( doubleEq( v, temp ) );

		v = reinterpret_cast< Arith* >(b.data() )->getOutput();
		MOOSE_ASSERT( doubleEq( v, temp2 ) );

		v = reinterpret_cast< Arith* >( c.data() )->getOutput();
		MOOSE_ASSERT( doubleEq( v, temp2 - temp ) );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		// Eref a( e1, i );
		// Eref b( e1, size - i - 1 );
		ObjId a( i1, i );
		ObjId b( i1, size - i - 1 );

		double temp = i;
		double temp2  = temp * temp;
		double ret = Field< double >::get( a, "x" );
		MOOSE_ASSERT( doubleEq( temp, ret ) );
		
		ret = Field< double >::get( b, "y" );
		MOOSE_ASSERT( doubleEq( temp2, ret ) );

		ret = Field< double >::get( a, "z" );
		MOOSE_ASSERT( doubleEq( temp2 - temp, ret ) );
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

	Element* elm = new GlobalDataElement( i2, ac, "test2", size );
	MOOSE_ASSERT( elm );
	ObjId obj( i2, 0 );

	Arith* arith = reinterpret_cast< Arith* >(obj.data() );
	for ( unsigned int i = 0; i < 4; ++i )
		arith->setIdentifiedArg( i, 0 );
	for ( unsigned int i = 0; i < 4; ++i )
		MOOSE_ASSERT( doubleEq( 0, arith->getIdentifiedArg( i ) ) );

	LookupField< unsigned int, double >::set( obj, "anyValue", 0, 100 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 1, 101 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 2, 102 );
	LookupField< unsigned int, double >::set( obj, "anyValue", 3, 103 );

	MOOSE_ASSERT( doubleEq( arith->getOutput(), 100 ) );
	MOOSE_ASSERT( doubleEq( arith->getArg1(), 101 ) );
	MOOSE_ASSERT( doubleEq( arith->getIdentifiedArg( 2 ), 102 ) );
	MOOSE_ASSERT( doubleEq( arith->getIdentifiedArg( 3 ), 103 ) );

	for ( unsigned int i = 0; i < 4; ++i )
		arith->setIdentifiedArg( i, 17 * i + 3 );

	double ret = LookupField< unsigned int, double >::get(
		obj, "anyValue", 0 );
	MOOSE_ASSERT( doubleEq( ret, 3 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 1 );
	MOOSE_ASSERT( doubleEq( ret, 20 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 2 );
	MOOSE_ASSERT( doubleEq( ret, 37 ) );

	ret = LookupField< unsigned int, double >::get( obj, "anyValue", 3 );
	MOOSE_ASSERT( doubleEq( ret, 54 ) );
	
	cout << "." << flush;
	i2.destroy();
}

void testIsA()
{
	const Cinfo* n = Neutral::initCinfo();
	const Cinfo* a = Arith::initCinfo();
	MOOSE_ASSERT( a->isA( "Arith" ) );
	MOOSE_ASSERT( a->isA( "Neutral" ) );
	MOOSE_ASSERT( !a->isA( "Fish" ) );
	MOOSE_ASSERT( !a->isA( "Synapse" ) );
	MOOSE_ASSERT( !n->isA( "Arith" ) );
	MOOSE_ASSERT( n->isA( "Neutral" ) );
	cout << "." << flush;
}

void testFinfoFields()
{
	const FinfoWrapper vmFinfo = IntFire::initCinfo()->findFinfo( "Vm" );
	const FinfoWrapper synFinfo = IntFire::initCinfo()->findFinfo( "synapse" );
	const FinfoWrapper procFinfo = IntFire::initCinfo()->findFinfo( "proc" );
	const FinfoWrapper processFinfo = IntFire::initCinfo()->findFinfo( "process" );
	const FinfoWrapper reinitFinfo = IntFire::initCinfo()->findFinfo( "reinit" );
	const FinfoWrapper spikeFinfo = IntFire::initCinfo()->findFinfo( "spikeOut" );
	const FinfoWrapper classNameFinfo = Neutral::initCinfo()->findFinfo( "className" );

	MOOSE_ASSERT( vmFinfo.getName() == "Vm" );
	MOOSE_ASSERT( vmFinfo.docs() == "Membrane potential" );
	MOOSE_ASSERT( vmFinfo.src().size() == 0 );
	MOOSE_ASSERT( vmFinfo.dest().size() == 2 );
	MOOSE_ASSERT( vmFinfo.dest()[0] == "setVm" );
	MOOSE_ASSERT( vmFinfo.dest()[1] == "getVm" );
	MOOSE_ASSERT( vmFinfo.type() == "double" );

	MOOSE_ASSERT( synFinfo.getName() == "synapse" );
	MOOSE_ASSERT( synFinfo.docs() == "Sets up field Elements for synapse" );
	MOOSE_ASSERT( synFinfo.src().size() == 0 );
	MOOSE_ASSERT( synFinfo.dest().size() == 0 );
	// cout <<  synFinfo->type() << endl;
	MOOSE_ASSERT( synFinfo.type() == typeid(Synapse).name() );

	MOOSE_ASSERT( procFinfo.getName() == "proc" );
	MOOSE_ASSERT( procFinfo.docs() == "Shared message for process and reinit" );
	MOOSE_ASSERT( procFinfo.src().size() == 0 );
	MOOSE_ASSERT( procFinfo.dest().size() == 2 );
	MOOSE_ASSERT( procFinfo.dest()[0] == "process" );
	MOOSE_ASSERT( procFinfo.dest()[1] == "reinit" );
	 // cout << "proc " << procFinfo.type() << endl;
	MOOSE_ASSERT( procFinfo.type() == "void" );
	
	MOOSE_ASSERT( processFinfo.getName() == "process" );
	MOOSE_ASSERT( processFinfo.docs() == "Handles process call" );
	MOOSE_ASSERT( processFinfo.src().size() == 0 );
	MOOSE_ASSERT( processFinfo.dest().size() == 0 );
	// cout << "process " << processFinfo.type() << endl;
	MOOSE_ASSERT( processFinfo.type() == "const ProcInfo*" );

	MOOSE_ASSERT( reinitFinfo.getName() == "reinit" );
	MOOSE_ASSERT( reinitFinfo.docs() == "Handles reinit call" );
	MOOSE_ASSERT( reinitFinfo.src().size() == 0 );
	MOOSE_ASSERT( reinitFinfo.dest().size() == 0 );
	// cout << "reinit " << reinitFinfo.type() << endl;
	MOOSE_ASSERT( reinitFinfo.type() == "const ProcInfo*" );

	MOOSE_ASSERT( spikeFinfo.getName() == "spikeOut" );
	MOOSE_ASSERT( spikeFinfo.docs() == "Sends out spike events" );
	MOOSE_ASSERT( spikeFinfo.src().size() == 0 );
	MOOSE_ASSERT( spikeFinfo.dest().size() == 0 );
	// cout << spikeFinfo->type() << endl;
	MOOSE_ASSERT( spikeFinfo.type() == "double" );

	MOOSE_ASSERT( classNameFinfo.getName() == "className" );
	MOOSE_ASSERT( classNameFinfo.type() == "string" );

	cout << "." << flush;
}

void testCinfoFields()
{
	MOOSE_ASSERT( IntFire::initCinfo()->getDocs() == "" );
	MOOSE_ASSERT( IntFire::initCinfo()->getBaseClass() == "SynHandler" );

	// We have a little bit of a hack here to cast away
	// constness, due to the way the FieldElementFinfos
	// are set up.
	Cinfo *neutralCinfo = const_cast< Cinfo* >( Neutral::initCinfo() );
	MOOSE_ASSERT( neutralCinfo->getNumSrcFinfo() == 1 );

	Cinfo *cinfo = const_cast< Cinfo* >( IntFire::initCinfo() );
	unsigned int nsf = neutralCinfo->getNumSrcFinfo();
	MOOSE_ASSERT( nsf == 1 );
	MOOSE_ASSERT( cinfo->getNumSrcFinfo() == 1 + nsf );
	MOOSE_ASSERT( cinfo->getSrcFinfo( 0 + nsf ) == cinfo->findFinfo( "spikeOut" ) );

	unsigned int ndf = neutralCinfo->getNumDestFinfo();
	MOOSE_ASSERT( ndf == 22 );
	unsigned int sdf = SynHandler::initCinfo()->getNumDestFinfo();
	MOOSE_ASSERT( sdf == 26 );
	MOOSE_ASSERT( cinfo->getNumDestFinfo() == 12 + sdf );

	MOOSE_ASSERT( cinfo->getDestFinfo( 0+ndf )->name() == "setNumSynapses" );
	MOOSE_ASSERT( cinfo->getDestFinfo( 1+ndf )->name() == "getNumSynapses" );
	MOOSE_ASSERT( cinfo->getDestFinfo( 2+ndf )->name() == "setNumSynapse" );
	MOOSE_ASSERT( cinfo->getDestFinfo( 3+ndf )->name() == "getNumSynapse" );

	MOOSE_ASSERT( cinfo->getDestFinfo( 0+sdf ) == cinfo->findFinfo( "setVm" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 1+sdf ) == cinfo->findFinfo( "getVm" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 2+sdf ) == cinfo->findFinfo( "setTau" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 3+sdf ) == cinfo->findFinfo( "getTau" ) );

	MOOSE_ASSERT( cinfo->getDestFinfo( 4+sdf ) == cinfo->findFinfo( "setThresh" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 5+sdf ) == cinfo->findFinfo( "getThresh" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 6+sdf ) == cinfo->findFinfo( "setRefractoryPeriod" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 7+sdf ) == cinfo->findFinfo( "getRefractoryPeriod" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 8+sdf ) == cinfo->findFinfo( "setBufferTime" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 9+sdf ) == cinfo->findFinfo( "getBufferTime" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 10+sdf ) == cinfo->findFinfo( "process" ) );
	MOOSE_ASSERT( cinfo->getDestFinfo( 11+sdf ) == cinfo->findFinfo( "reinit" ) );


	unsigned int nvf = neutralCinfo->getNumValueFinfo();
	MOOSE_ASSERT( nvf == 14 );
	MOOSE_ASSERT( cinfo->getNumValueFinfo() == 6 + nvf );
	MOOSE_ASSERT( cinfo->getValueFinfo( 0 + nvf ) == cinfo->findFinfo( "numSynapses" ) );
	MOOSE_ASSERT( cinfo->getValueFinfo( 1 + nvf ) == cinfo->findFinfo( "Vm" ) );
	MOOSE_ASSERT( cinfo->getValueFinfo( 2 + nvf ) == cinfo->findFinfo( "tau" ) );
	MOOSE_ASSERT( cinfo->getValueFinfo( 3 + nvf ) == cinfo->findFinfo( "thresh" ) );
	MOOSE_ASSERT( cinfo->getValueFinfo( 4 + nvf ) == cinfo->findFinfo( "refractoryPeriod" ) );
	MOOSE_ASSERT( cinfo->getValueFinfo( 5 + nvf ) == cinfo->findFinfo( "bufferTime" ) );

	unsigned int nlf = neutralCinfo->getNumLookupFinfo();
	MOOSE_ASSERT( nlf == 3 ); // Neutral inserts a lookup field for neighbours
	MOOSE_ASSERT( cinfo->getNumLookupFinfo() == 0 + nlf );
	MOOSE_ASSERT( cinfo->getLookupFinfo( 0 + nlf )->name() == "dummy");

	unsigned int nshf = neutralCinfo->getNumSharedFinfo();
	MOOSE_ASSERT( nshf == 0 );
	MOOSE_ASSERT( cinfo->getNumSharedFinfo() == 1 + nshf );
	MOOSE_ASSERT( cinfo->getSharedFinfo( 0 + nshf ) == cinfo->findFinfo( "proc" ) );

	cout << "." << flush;
}

void testCinfoElements()
{
	Id intFireCinfoId( "/classes/IntFire" );
	// const Cinfo *neutralCinfo = Neutral::initCinfo();
	// unsigned int nvf = neutralCinfo->getNumValueFinfo();
	// unsigned int nsf = neutralCinfo->getNumSrcFinfo();
	// unsigned int ndf = neutralCinfo->getNumDestFinfo();
	//unsigned int sdf = SynHandler::initCinfo()->getNumDestFinfo();

	MOOSE_ASSERT( intFireCinfoId != Id() );
	MOOSE_ASSERT( Field< string >::get( intFireCinfoId, "name" ) == "IntFire" );
	MOOSE_ASSERT( Field< string >::get( intFireCinfoId, "baseClass" ) == 
					"SynHandler" );
	Id intFireValueFinfoId( "/classes/IntFire/valueFinfo" );
	unsigned int n = Field< unsigned int >::get( 
		intFireValueFinfoId, "numData" );
	MOOSE_ASSERT( n == 5 );
	Id intFireSrcFinfoId( "/classes/IntFire/srcFinfo" );
	MOOSE_ASSERT( intFireSrcFinfoId != Id() );
	n = Field< unsigned int >::get( intFireSrcFinfoId, "numData" );
	MOOSE_ASSERT( n == 1 );
	Id intFireDestFinfoId( "/classes/IntFire/destFinfo" );
	MOOSE_ASSERT( intFireDestFinfoId != Id() );
	n = Field< unsigned int >::get( intFireDestFinfoId, "numData" );
	MOOSE_ASSERT( n == 12 );
	
	ObjId temp( intFireSrcFinfoId, 0 );
	string foo = Field< string >::get( temp, "fieldName" );
	MOOSE_ASSERT( foo == "spikeOut" );

	foo = Field< string >::get( temp, "type" );
	MOOSE_ASSERT( foo == "double" );

	n = Field< unsigned int >::get( intFireDestFinfoId, "numField" );
	MOOSE_ASSERT( n == 1 );

	temp = ObjId( intFireDestFinfoId, 7 );
	string str = Field< string >::get( temp, "fieldName" );
	MOOSE_ASSERT( str == "getRefractoryPeriod");
	temp = ObjId( intFireDestFinfoId, 11 );
	str = Field< string >::get( temp, "fieldName" );
	MOOSE_ASSERT( str == "reinit" );
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
	Element* e1 = new GlobalDataElement( t1, Test::initCinfo(), "test1" );
	MOOSE_ASSERT( e1 );
	MOOSE_ASSERT( e1 == t1.element() );
	Element* e2 = new GlobalDataElement( t2, Test::initCinfo(), "test2", 1 );
	// ret = Test::initCinfo()->create( t2, "test2", 1 );
	MOOSE_ASSERT( e2 );
	MOOSE_ASSERT( e2 == t2.element() );

	// Set up message. The actual routine is in Shell.cpp, but here we
	// do it independently.
	const Finfo* shareFinfo = Test::initCinfo()->findFinfo( "shared" );
	MOOSE_ASSERT( shareFinfo != 0 );
	Msg* m = new OneToOneMsg( t1.eref(), t2.eref(), 0 );
	MOOSE_ASSERT( m != 0 );
	bool ret = shareFinfo->addMsg( shareFinfo, m->mid(), t1.element() );
	MOOSE_ASSERT( ret );

	//////////////////////////////////////////////////////////////
	// Test Element::getFieldsOfOutgoingMsg
	//////////////////////////////////////////////////////////////
	vector< pair< BindIndex, FuncId > > pairs;
	e1->getFieldsOfOutgoingMsg( m->mid(), pairs );
	MOOSE_ASSERT( pairs.size() == 3 );
	MOOSE_ASSERT( pairs[0].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[0])->getBindIndex() );
	MOOSE_ASSERT( pairs[0].second == dynamic_cast< DestFinfo* >(Test::sharedVec[1])->getFid() );

	MOOSE_ASSERT( pairs[1].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[2])->getBindIndex() );
	MOOSE_ASSERT( pairs[1].second == dynamic_cast< DestFinfo* >(Test::sharedVec[3])->getFid() );

	MOOSE_ASSERT( pairs[2].first == dynamic_cast< SrcFinfo* >(Test::sharedVec[4])->getBindIndex() );
	MOOSE_ASSERT( pairs[2].second == dynamic_cast< DestFinfo* >(Test::sharedVec[5])->getFid() );

	e2->getFieldsOfOutgoingMsg( m->mid(), pairs );
	MOOSE_ASSERT( pairs.size() == 3 );

	//////////////////////////////////////////////////////////////
	// Test Cinfo::srcFinfoName
	//////////////////////////////////////////////////////////////
	MOOSE_ASSERT( Test::initCinfo()->srcFinfoName( pairs[0].first ) == "s0" );
	MOOSE_ASSERT( Test::initCinfo()->srcFinfoName( pairs[1].first ) == "s1" );
	MOOSE_ASSERT( Test::initCinfo()->srcFinfoName( pairs[2].first ) == "s2" );

	//////////////////////////////////////////////////////////////
	// Test Cinfo::destFinfoName
	//////////////////////////////////////////////////////////////
	MOOSE_ASSERT( Test::initCinfo()->destFinfoName( pairs[0].second ) == "d0" );
	MOOSE_ASSERT( Test::initCinfo()->destFinfoName( pairs[1].second ) == "d1" );
	MOOSE_ASSERT( Test::initCinfo()->destFinfoName( pairs[2].second ) == "d2" );
	//////////////////////////////////////////////////////////////
	// Test Msg::getSrcFieldsOnE1 and family
	//////////////////////////////////////////////////////////////
	vector< string > fieldNames;
	fieldNames = m->getSrcFieldsOnE1();
	MOOSE_ASSERT( fieldNames.size() == 3 );
	MOOSE_ASSERT( fieldNames[0] == "s0" );
	MOOSE_ASSERT( fieldNames[1] == "s1" );
	MOOSE_ASSERT( fieldNames[2] == "s2" );

	fieldNames = m->getDestFieldsOnE2();
	MOOSE_ASSERT( fieldNames.size() == 3 );
	MOOSE_ASSERT( fieldNames[0] == "d0" );
	MOOSE_ASSERT( fieldNames[1] == "d1" );
	MOOSE_ASSERT( fieldNames[2] == "d2" );

	fieldNames = m->getSrcFieldsOnE2();
	MOOSE_ASSERT( fieldNames.size() == 3 );
	MOOSE_ASSERT( fieldNames[0] == "s0" );
	MOOSE_ASSERT( fieldNames[1] == "s1" );
	MOOSE_ASSERT( fieldNames[2] == "s2" );

	fieldNames = m->getDestFieldsOnE1();
	MOOSE_ASSERT( fieldNames.size() == 3 );
	MOOSE_ASSERT( fieldNames[0] == "d0" );
	MOOSE_ASSERT( fieldNames[1] == "d1" );
	MOOSE_ASSERT( fieldNames[2] == "d2" );

	//////////////////////////////////////////////////////////////
	// getMsgTargetAndFunctions
	//////////////////////////////////////////////////////////////
	vector< ObjId > tgt;
	vector< string > func;
	unsigned int numTgt = e1->getMsgTargetAndFunctions( 0, 
					dynamic_cast< SrcFinfo* >(Test::sharedVec[0] ),
					tgt, func );
	MOOSE_ASSERT( numTgt == tgt.size() );
	MOOSE_ASSERT( tgt.size() == 1 );
	MOOSE_ASSERT( tgt[0] == ObjId( t2, 0 ) );
	MOOSE_ASSERT( func[0] == "d0" );

	// Note that the srcFinfo #2 is in sharedVec[4]
	numTgt = e2->getMsgTargetAndFunctions( 0, 
					dynamic_cast< SrcFinfo* >(Test::sharedVec[4] ),
					tgt, func );
	MOOSE_ASSERT( numTgt == tgt.size() );
	MOOSE_ASSERT( tgt.size() == 1 );
	MOOSE_ASSERT( tgt[0] == ObjId( t1, 0 ) );
	MOOSE_ASSERT( func[0] == "d2" );

	//////////////////////////////////////////////////////////////
	// Clean up.
	//////////////////////////////////////////////////////////////
	t1.destroy();
	t2.destroy();
	cout << "." << flush;
}

void testHopFunc()
{
	extern const double* checkHopFuncTestBuffer();

	HopIndex hop2( 1234, MooseTestHop );
	HopFunc2< string, double > two( hop2 );

	two.op( Id(3).eref(), "two", 2468.0 );
	const double* buf = checkHopFuncTestBuffer();
	const TgtInfo* tgt = reinterpret_cast< const TgtInfo* >( buf );
	MOOSE_ASSERT( tgt->bindIndex() == 1234 );
	MOOSE_ASSERT( tgt->dataSize() == 2 );
	const char* c = reinterpret_cast< const char* >( 
					buf + TgtInfo::headerSize );
	MOOSE_ASSERT( strcmp( c, "two" ) == 0 );
	MOOSE_ASSERT( doubleEq( buf[TgtInfo::headerSize + 1], 2468.0 ) );

	HopIndex hop3( 36912, MooseTestHop );
	HopFunc3< string, int, vector< double > > three( hop3 );
	vector< double > temp( 3 );
	temp[0] = 11222;
	temp[1] = 24332;
	temp[2] = 234232342;
	three.op( Id(3).eref(), "three", 3333, temp );

	MOOSE_ASSERT( tgt->bindIndex() == 36912 );
	MOOSE_ASSERT( tgt->dataSize() == 6 );
	c = reinterpret_cast< const char* >( buf + TgtInfo::headerSize );
	MOOSE_ASSERT( strcmp( c, "three" ) == 0 );
	int i = TgtInfo::headerSize + 1;
	MOOSE_ASSERT( doubleEq( buf[i++], 3333.0 ) );
	MOOSE_ASSERT( doubleEq( buf[i++], 3.0 ) ); // Size of array.
	MOOSE_ASSERT( doubleEq( buf[i++], temp[0] ) );
	MOOSE_ASSERT( doubleEq( buf[i++], temp[1] ) );
	MOOSE_ASSERT( doubleEq( buf[i++], temp[2] ) );

	cout << "." << flush;
}

void testAsync( )
{
	showFields();
	testSendMsg();
	testCreateMsg();
	testSetGet();
	testSetGetDouble();
	testSetGetSynapse();
	testSetGetVec();
	test2ArgSetVec();
	testSetRepeat();
	testStrSet();
	testStrGet();
	testLookupSetGet();
//	testSendSpike();
	testSparseMatrix();
	testSparseMatrix2();
	testSparseMatrixReorder();
	testSparseMatrixFill();
	testSparseMsg();
	testSharedMsg();
	testConvVector();
	testConvVectorOfVectors();
	testMsgField();
	// testSetGetExtField(); Unsure if we're keeping ext fields.
	testIsA();
	testFinfoFields();
	testCinfoFields();
	testCinfoElements();
	testMsgSrcDestFields();
	testHopFunc();
}
