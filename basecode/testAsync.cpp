/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Neutral.h"
#include "Dinfo.h"
#include "Shell.h"
#include "Message.h"
#include <queue>
#include "../biophysics/Synapse.h"
#include "../biophysics/IntFire.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "../randnum/randnum.h"
#include "../scheduling/Tick.h"
#include "../scheduling/TickPtr.h"
#include "../scheduling/Clock.h"

void insertIntoQ( )
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;

	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new SingleMsg( e1, e2 );
	

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "objname_%d", i );
		string stemp( temp );
		char buf[200];

		// This simulates a sendTo
		unsigned int size = Conv< string >::val2buf( buf, stemp );
		Qinfo qi( 1, i, size + sizeof( DataId ), 1, 1 );

		*reinterpret_cast< DataId* >( buf + size ) = DataId( i );

		qi.addToQ( 0, m->mid(), 1, buf );
	}
	Qinfo::clearQ( 0 );

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "objname_%d", i );
		string name = ( reinterpret_cast< Neutral* >(e2.element()->data( i )) )->getName();
		assert( name == temp );
	}
	cout << "." << flush;

	delete m;
	delete i1();
	delete i2();
}

void testSendMsg()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	FuncId fid = 1;

	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	Msg* m = new OneToOneMsg( e1.element(), e2.element() );
	// Conn c;
	// c.add( m );
	ConnId cid = 0;
	e1.element()->addMsgToConn( m->mid(), cid );
	
	SrcFinfo1<string> s( "test", "", cid );
	s.registerSrcFuncIndex( 0 );
	e1.element()->addTargetFunc( fid, s.getFuncIndex() );

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "send_to_e2_%d", i );
		string stemp( temp );
		s.send( Eref( e1.element(), i ), stemp );
	}
	Qinfo::clearQ( 0 );

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "send_to_e2_%d", i );
		assert( reinterpret_cast< Neutral* >(e2.element()->data( i ))->getName()
			== temp );
	}
	cout << "." << flush;

	delete i1();
	delete i2();
}

void testCreateMsg()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	Id i1 = nc->create( "test1", size );
	Id i2 = nc->create( "test2", size );

	Eref e1 = i1.eref();
	Eref e2 = i2.eref();

	bool ret = add( e1.element(), "child", e2.element(), "parent" );
	
	assert( ret );

	const Finfo* f = nc->findFinfo( "child" );

	for ( unsigned int i = 0; i < size; ++i ) {
		const SrcFinfo0* sf = dynamic_cast< const SrcFinfo0* >( f );
		assert( sf != 0 );
		sf->send( Eref( e1.element(), i ) );
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

void testSet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );

	Eref e2 = i2.eref();
	
	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "set_e2_%d", i );
		string stemp( temp );
		Eref dest( e2.element(), i );
		set( dest, "set_name", stemp );
		Qinfo::clearQ( 0 );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "set_e2_%d", i );
		assert( reinterpret_cast< Neutral* >(e2.element()->data( i ))->getName()
			== temp );
	}

	cout << "." << flush;

	delete i2();
}

void testGet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );
	Element* shell = Id()();

	Eref e2 = i2.eref();
	
	for ( unsigned int i = 0; i < size; ++i ) {
		char temp[20];
		sprintf( temp, "get_e2_%d", i );
		string stemp( temp );
		reinterpret_cast< Neutral* >(e2.element()->data( i ))->setName( temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		string stemp;
		Eref dest( e2.element(), i );

			// I don't really want an array of SetGet/Shells to originate
			// get requests, but just
			// to test that it works from anywhere...
		if ( get( dest, "get_name" ) ) {
			Qinfo::clearQ( 0 ); // Request goes to e2
			// shell->clearQ(); // Response comes back to e1

			stemp = ( reinterpret_cast< Shell* >(shell->data( 0 )) )->getBuf();
			// cout << i << "	" << stemp << endl;
			char temp[20];
			sprintf( temp, "get_e2_%d", i );
			assert( stemp == temp );
		}
	}

	cout << "." << flush;
	delete i2();
}

void testSetGet()
{
	const Cinfo* nc = Neutral::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = nc->create( "test2", size );

	
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		char temp[20];
		sprintf( temp, "sg_e2_%d", i );
		bool ret = SetGet1< string >::set( e2, "name", temp );
		assert( ret );
		assert( reinterpret_cast< Neutral* >(e2.data())->getName() == temp );
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		char temp[20];
		sprintf( temp, "sg_e2_%d", i );
		string ret = SetGet1< string >::get( e2, "name" );
		assert( ret == temp );
	}

	cout << "." << flush;
	delete i2();
}

void testSetGetDouble()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = ic->create( "test2", size );

	
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double temp = i;
		bool ret = SetGet1< double >::set( e2, "Vm", temp );
		assert( ret );
		assert( 
			fabs ( reinterpret_cast< IntFire* >(e2.data())->getVm() - temp ) <
				EPSILON ); 
	}

	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		double temp = i;
		double ret = SetGet1< double >::get( e2, "Vm" );
		assert( fabs ( temp - ret ) < EPSILON );
	}

	cout << "." << flush;
	delete i2();
}

void testSetGetSynapse()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = ic->create( "test2", size );
	// SynElement syn( sc, i2() );
	FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );

	assert( syn.numData() == 0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref e2( i2(), i );
		bool ret = SetGet1< unsigned int >::set( e2, "numSynapses", i );
		assert( ret );
	}
	assert( syn.numData() == ( size * (size - 1) ) / 2 );
	// cout << "NumSyn = " << syn.numData() << endl;
	
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j );
			Eref syne( &syn, di );
			double temp = i * 1000 + j ;
			bool ret = SetGet1< double >::set( syne, "delay", temp );
			assert( ret );
			assert( 
			fabs ( reinterpret_cast< Synapse* >(syne.data())->getDelay() - temp ) <
				EPSILON ); 
		}
	}
	cout << "." << flush;
	delete i2();
}

void testSetGetVec()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = ic->create( "test2", size );
//	SynElement syn( sc, i2() );
	FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );

	assert( syn.numData() == 0 );
	vector< unsigned int > numSyn( size, 0 );
	for ( unsigned int i = 0; i < size; ++i )
		numSyn[i] = i;
	
	Eref e2( i2(), 0 );
	// Here we test setting a 1-D vector
	bool ret = SetGet1< unsigned int >::setVec( e2, "numSynapses", numSyn );
	assert( ret );
	unsigned int nd = syn.numData();
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

	Eref se( &syn, 0 );
	ret = SetGet1< double >::setVec( se, "delay", delay );
	for ( unsigned int i = 0; i < size; ++i ) {
		for ( unsigned int j = 0; j < i; ++j ) {
			DataId di( i, j );
			Eref syne( &syn, di );
			double temp = i * 1000 + j ;
			assert( 
			fabs ( reinterpret_cast< Synapse* >(syne.data())->getDelay() - temp ) <
				EPSILON ); 
		}
	}
	cout << "." << flush;
	delete i2();
}

void testSendSpike()
{
	static const double EPSILON = 1e-9;
	const Cinfo* ic = IntFire::initCinfo();
	const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 100;
	string arg;
	Id i2 = ic->create( "test2", size );
	Eref e2 = i2.eref();
	//SynElement syn( sc, i2() );
	FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );

	assert( syn.numData() == 0 );
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref er( i2(), i );
		bool ret = SetGet1< unsigned int >::set( er, "numSynapses", i );
		assert( ret );
	}
	assert( syn.numData() == ( size * (size - 1) ) / 2 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( &syn, di );

	bool ret = SingleMsg::add( e2, "spike", syne, "addSpike" );
	assert( ret );

	ret = SetGet1< double >::set( e2, "Vm", 1.0 );
	ProcInfo p;
	reinterpret_cast< IntFire* >(e2.data())->process( &p, e2 );
	Qinfo::clearQ( 0 );
//	e2.element()->clearQ();
	Eref synParent( e2.element(), 1 );

	reinterpret_cast< IntFire* >(synParent.data())->process( &p, synParent);
	double Vm = SetGet1< double >::get( synParent, "Vm" );
	assert( fabs( Vm + 1e-7) < EPSILON );
	// cout << "Vm = " << Vm << endl;
	cout << "." << flush;
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
		double Vm = SetGet1< double >::get( er, field );
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
		{ 20112, 240, 144, 432, 864, 2016, 3600, 4704, 6192, 7248 };
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
	const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 1024;
	string arg;
	Id i2 = ic->create( "test2", size );
	Eref e2 = i2.eref();
	// SynElement syn( sc, i2() );
	FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );

	assert( syn.numData() == 0 );
	/*
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref er( i2(), i );
		bool ret = SetGet1< unsigned int >::set( er, "numSynapses", i );
		assert( ret );
	}
	assert( syn.numData() == ( size * (size - 1) ) / 2 );
	*/

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( &syn, di );

	bool ret = SparseMsg::add( e2.element(), "spike", &syn, "addSpike", 
		connectionProbability );
	assert( ret );

	unsigned int nd = syn.numData();
	// cout << "Num Syn = " << nd << endl;
	assert( nd == NUMSYN );
	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	ret = SetGet1< double >::setVec( e2, "Vm", temp );
	assert( ret );
	temp.clear();
	temp.resize( size, thresh );
	ret = SetGet1< double >::setVec( e2, "thresh", temp );
	assert( ret );
	temp.clear();
	temp.resize( size, refractoryPeriod );
	ret = SetGet1< double >::setVec( e2, "refractoryPeriod", temp );
	assert( ret );

	vector< double > weight;
	weight.reserve( nd );
	vector< double > delay;
	delay.reserve( nd );
	for ( unsigned int i = 0; i < size; ++i ) {
		unsigned int numSyn = syne.element()->numData2( i );
		for ( unsigned int j = 0; j < numSyn; ++j ) {
			weight.push_back( mtrand() * weightMax );
			delay.push_back( mtrand() * delayMax );
		}
	}
	ret = SetGet1< double >::setVec( syne, "weight", weight );
	assert( ret );
	ret = SetGet1< double >::setVec( syne, "delay", delay );
	assert( ret );

	// printGrid( i2(), "Vm", 0, thresh );

	ProcInfo p;
	p.dt = timestep;

	for ( unsigned int i = 0; i < runsteps; ++i ) {
		p.currTime += p.dt;
		i2()->process( &p );
		assert( syn.q_.size() == qSize[i] );
		// cout << "T = " << p.currTime << ", Q size = " << syn.q_.size() << endl;
		Qinfo::clearQ( 0 );
//		i2()->process( &p );
//		printGrid( i2(), "Vm", 0, thresh );
		// sleep(1);
	}
	// printGrid( i2(), "Vm", 0, thresh );

	cout << "." << flush;
	delete i2();
}

void testUpValue()
{
	static const double EPSILON = 1e-9;
	const Cinfo* cc = Clock::initCinfo();
	const Cinfo* tc = Tick::initCinfo();
	unsigned int size = 10;
	Id clock = cc->create( "clock", 1 );
	Eref clocker = clock.eref();
	//SynElement syn( sc, i2() );
	FieldElement< Tick, Clock, &Clock::getTick > ticke( tc, clock(), &Clock::getNumTicks, &Clock::setNumTicks );

	assert( ticke.numData() == 0 );
	bool ret = SetGet1< unsigned int >::set( clocker, "numTicks", size );
	assert( ret );
	assert( ticke.numData() == size );


	for ( unsigned int i = 0; i < size; ++i ) {
		DataId di( 0, i ); // DataId( data, field )
		Eref te( &ticke, di );
		double dt = i;
		ret = SetGet1< double >::set( te, "dt", dt );
		assert( ret );
		double val = SetGet1< double >::get( te, "localdt" );
		assert( fabs( dt - val ) < EPSILON );

		dt *= 10.0;
		ret = SetGet1< double >::set( te, "localdt", dt );
		assert( ret );
		val = SetGet1< double >::get( te, "dt" );
		assert( fabs( dt - val ) < EPSILON );
	}
	cout << "." << flush;
	delete clock();
}

void testAsync( )
{
	insertIntoQ();
	testSendMsg();
	testCreateMsg();
	testSet();
	testGet();
	testSetGet();
	testSetGetDouble();
	testSetGetSynapse();
	testSetGetVec();
	testSendSpike();
	testSparseMatrix();
	testSparseMsg();
	testUpValue();
}
