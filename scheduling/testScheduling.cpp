/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Tick.h"
#include "TickPtr.h"
#include "Clock.h"
#include "testScheduling.h"

#include <queue>
#include "../biophysics/Synapse.h"
#include "../biophysics/IntFire.h"
#include "MsgManager.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "../randnum/randnum.h"

pthread_mutex_t TestSched::mutex_;
bool TestSched::isInitPending_( 1 );
int TestSched::globalIndex_( 0 );

//////////////////////////////////////////////////////////////////////
// Setting up a class for testing scheduling.
//////////////////////////////////////////////////////////////////////

static DestFinfo processFinfo( "process",
	"handles process call",
	new EpFunc1< TestSched, ProcPtr>( &TestSched::eprocess ) );
const Cinfo* TestSched::initCinfo()
{

	static Finfo* testSchedFinfos[] = {
		&processFinfo
	};

	static Cinfo testSchedCinfo (
		"testSched",
		0,
		testSchedFinfos,
		sizeof ( testSchedFinfos ) / sizeof( Finfo* ),
		new Dinfo< TestSched >()
	);

	return &testSchedCinfo;
}

static const Cinfo* testSchedCinfo = TestSched::initCinfo();

void TestSched::process( const ProcInfo*p, const Eref& e )
{
	static const int timings[] = { 1, 2, 2, 2, 3, 3, 4, 4, 4, 
		5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10,
		11, 12, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15,
		16, 16, 16, 17, 18, 18, 18, 18, 19, 20, 20, 20, 20, 20 };
	// unsigned int max = sizeof( timings ) / sizeof( int );
	// cout << Shell::myNode() << ":" << p->threadIndexInGroup << " : timing[ " << index_ << ", " << p->threadId << " ] = " << timings[ index_ ] << ", time = " << p->currTime << endl;
	if ( static_cast< int >( p->currTime ) != timings[ index_ ] ) {
		cout << Shell::myNode() << ":" << p->threadIndexInGroup << " :testThreadSchedElement::process: index= " << index_ << ", numThreads = " <<
			p->numThreads << ", currTime = " << p->currTime << endl;
	}

	assert( static_cast< int >( p->currTime ) == timings[ index_ ] );
	++index_;
	/*
	assert( static_cast< int >( p->currTime ) == 	
		timings[ index_ / p->numThreadsInGroup ] );
		*/

	// Check that everything remains in sync across threads.
	pthread_mutex_lock( &mutex_ );
		assert( ( globalIndex_ - index_ )*( globalIndex_ - index_ ) <= 1 );
		if ( p->threadIndexInGroup == 0 )
			globalIndex_ = index_;
	pthread_mutex_unlock( &mutex_ );

	// assert( index_ <= max * p->numThreads );
	// cout << index_ << ": " << p->currTime << endl;
}

//////////////////////////////////////////////////////////////////////

/**
 * Check that the ticks are set up properly, created and destroyed as
 * needed, and are sorted when dts are assigned
 */
void setupTicks()
{
	static const double EPSILON = 1.0e-9;
	const double runtime = 20.0;
	// const Cinfo* tc = Tick::initCinfo();
	Id clock = Id::nextId();
	vector< unsigned int > dims( 1, 1 );
	Element* clocke = new Element( clock, Clock::initCinfo(), "tclock",
		dims, 1 );
	assert( clocke );
	// bool ret = Clock::initCinfo()->create( clock, "tclock", 1 );
	// assert( ret );
	// Element* clocke = clock();
	Eref clocker = clock.eref();
	Id tickId( clock.value() + 1 );
	Element* ticke = tickId();
	assert( ticke->name() == "tick" );

	// FieldElement< Tick, Clock, &Clock::getTick > ticke( tc, clocke, &Clock::getNumTicks, &Clock::setNumTicks );
	unsigned int size = 10;

	// bool ret = OneToAllMsg::add( clocker, "childTick", ticke, "parent" );
	// assert( ret );

	assert( ticke->dataHandler()->numData() == 0 );

	bool ret = Field< unsigned int >::set( clocker, "numTicks", size );
	Clock* clockData = reinterpret_cast< Clock* >( clocker.data() );
	clockData->setNumTicks( size );

	// assert( ret );
	// cout << Shell::myNode() << ": numTicks: " << ticke->dataHandler()->numData() << ", " << size << endl;
	assert( ticke->dataHandler()->numData() == size );

	Eref er0( ticke, DataId( 0, 2 ) );
	ret = Field< double >::set( er0, "dt", 5.0);
	assert( ret );
	ret = Field< unsigned int >::set( er0, "stage", 0);
	assert( ret );
	Eref er1( ticke, DataId( 0, 1 ) );
	ret = Field< double >::set( er1, "dt", 2.0);
	assert( ret );
	ret = Field< unsigned int >::set( er1, "stage", 0);
	assert( ret );
	Eref er2( ticke, DataId( 0, 0 ) );
	ret = Field< double >::set( er2, "dt", 2.0);
	assert( ret );
	ret = Field< unsigned int >::set( er2, "stage", 1);
	assert( ret );
	Eref er3( ticke, DataId( 0, 3 ) );
	ret = Field< double >::set( er3, "dt", 1.0);
	assert( ret );
	ret = Field< unsigned int >::set( er3, "stage", 0);
	assert( ret );
	Eref er4( ticke, DataId( 0, 4 ) );
	ret = Field< double >::set( er4, "dt", 3.0);
	assert( ret );
	ret = Field< unsigned int >::set( er4, "stage", 5);
	assert( ret );
	// Note that here I put the tick on a different DataId. later it gets
	// to sit on the appropriate Conn, when the SingleMsg is set up.
	Eref er5( ticke, DataId( 0, 7 ) );
	ret = Field< double >::set( er5, "dt", 5.0);
	assert( ret );
	ret = Field< unsigned int >::set( er5, "stage", 1);
	assert( ret );

	Clock* cdata = reinterpret_cast< Clock* >( clocker.data() );
	assert( cdata->tickPtr_.size() == 4 );
	assert( fabs( cdata->tickPtr_[0].dt_ - 1.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[1].dt_ - 2.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[2].dt_ - 3.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[3].dt_ - 5.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[0].nextTime_ - 1.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[1].nextTime_ - 2.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[2].nextTime_ - 3.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[3].nextTime_ - 5.0 ) < EPSILON );
	assert( cdata->tickPtr_[0].ticks_.size() == 1 );
	assert( cdata->tickPtr_[1].ticks_.size() == 2 );
	assert( cdata->tickPtr_[2].ticks_.size() == 1 );
	assert( cdata->tickPtr_[3].ticks_.size() == 2 );

	assert( cdata->tickPtr_[0].ticks_[0] == reinterpret_cast< const Tick* >( er3.data() ) );
	assert( cdata->tickPtr_[1].ticks_[0] == reinterpret_cast< const Tick* >( er1.data() ) );
	assert( cdata->tickPtr_[1].ticks_[1] == reinterpret_cast< const Tick* >( er2.data() ) );
	assert( cdata->tickPtr_[2].ticks_[0] == reinterpret_cast< const Tick* >( er4.data() ) );
	assert( cdata->tickPtr_[3].ticks_[0] == reinterpret_cast< const Tick* >( er0.data() ) );
	assert( cdata->tickPtr_[3].ticks_[1] == reinterpret_cast< const Tick* >( er5.data() ) );

	Id tsid = Id::nextId();
	Element* tse = new Element( tsid, testSchedCinfo, "tse", dims, 1 );

	// testSchedElement tse;
	Eref ts( tse, 0 );
	
	// No idea what FuncId to use here. Assume 0.
	FuncId f( processFinfo.getFid() );
	SingleMsg *m0 = new SingleMsg( er0, ts ); 
	er0.element()->addMsgAndFunc( m0->mid(), f, 0 );
	SingleMsg *m1 = new SingleMsg( er1, ts ); 
	er1.element()->addMsgAndFunc( m1->mid(), f, 1 );
	SingleMsg *m2 = new SingleMsg( er2, ts );
	er2.element()->addMsgAndFunc( m2->mid(), f, 2 );
	SingleMsg *m3 = new SingleMsg( er3, ts ); 
	er3.element()->addMsgAndFunc( m3->mid(), f, 3 );
	SingleMsg *m4 = new SingleMsg( er4, ts ); 
	er4.element()->addMsgAndFunc( m4->mid(), f, 4 );
	SingleMsg *m5 = new SingleMsg( er5, ts ); 
	er5.element()->addMsgAndFunc( m5->mid(), f, 7 );

	cdata->rebuild();

	Qinfo q( 0, 0, 8 ); // Not really used in the 'start' function.
	cdata->start( clocker, &q, runtime );

	assert( fabs( cdata->getCurrentTime() - runtime ) < 1e-6 );

	tickId.destroy();
	clock.destroy();
	tsid.destroy();
	// tickId.destroy();
	// cout << "done setupTicks\n";
	cout << "." << flush;
}

void testThreads()
{
	Element* se = Id()();
	Shell* s = reinterpret_cast< Shell* >( se->dataHandler()->data( 0 ) );
	s->setclock( 0, 5.0, 0 );
	s->setclock( 1, 2.0, 0 );
	s->setclock( 2, 2.0, 1 );
	s->setclock( 3, 1.0, 0 );
	s->setclock( 4, 3.0, 5 );
	s->setclock( 5, 5.0, 1 );

	vector< unsigned int > dims;
	dims.push_back( 7 ); // A suitable number to test dispatch of Process calls during threading.
	Id tsid = Id::nextId();
	Element* tse = new Element( tsid, testSchedCinfo, "tse", dims, 1 );
	// testThreadSchedElement tse;
	Eref ts( tse, 0 );
	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );
	Eref er1( ticke, DataId( 0, 1 ) );
	Eref er2( ticke, DataId( 0, 2 ) );
	Eref er3( ticke, DataId( 0, 3 ) );
	Eref er4( ticke, DataId( 0, 4 ) );
	Eref er5( ticke, DataId( 0, 5 ) );

	FuncId f( processFinfo.getFid() );
	SingleMsg* m0 = new SingleMsg( er0, ts );
	er0.element()->addMsgAndFunc( m0->mid(), f, 0 );
	SingleMsg* m1 = new SingleMsg( er1, ts );
	er1.element()->addMsgAndFunc( m1->mid(), f, 1 );
	SingleMsg* m2 = new SingleMsg( er2, ts );
	er2.element()->addMsgAndFunc( m2->mid(), f, 2 );
	SingleMsg* m3 = new SingleMsg( er3, ts );
	er3.element()->addMsgAndFunc( m3->mid(), f, 3 );
	SingleMsg* m4 = new SingleMsg( er4, ts );
	er4.element()->addMsgAndFunc( m4->mid(), f, 4 );
	SingleMsg* m5 = new SingleMsg( er5, ts );
	er5.element()->addMsgAndFunc( m5->mid(), f, 5 );
	s->start( 10 );

	// Qinfo::mergeQ( 0 ); // Need to clean up stuff.

	// cout << "Done TestThreads" << flush;
	tsid.destroy();
	cout << "." << flush;
}

void testThreadIntFireNetwork()
{
	// Known value from single-thread run, at t = 1 sec.
	// These are the old values
	static const double Vm100 = 0.0857292;
	static const double Vm900 = 0.107449;
	// static const double Vm100 = 0.10124059893763067;
	// static const double Vm900 = 0.091409481280996352;
	static const unsigned int NUMSYN = 104576;
	static const double thresh = 0.2;
	static const double Vmax = 1.0;
	static const double refractoryPeriod = 0.4;
	static const double weightMax = 0.02;
	static const double delayMax = 4;
	static const double timestep = 0.2;
	static const double connectionProbability = 0.1;
	static const unsigned int runsteps = 5;
	// static const unsigned int runsteps = 1000;
	const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 1024;
	string arg;

	// Qinfo::mergeQ( 0 );

	mtseed( 5489UL ); // The default value, but better to be explicit.

	Id i2 = Id::nextId();
	// bool ret = ic->create( i2, "test2", size );
	vector< unsigned int > dims( 1, size );
	Element* t2 = new Element( i2, ic, "test2", dims );
	assert( t2 );

	Eref e2 = i2.eref();
	// FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert( syn->name() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( syn, di );

	unsigned int numThreads = 1;
	if ( Qinfo::numSimGroup() >= 2 ) {
		numThreads = Qinfo::simGroup( 1 )->numThreads;
	}
	/*
	bool ret = SparseMsg::add( e2.element(), "spike", syn, "addSpike", 
		connectionProbability, numThreads ); // Include group id as an arg. 
	assert( ret );
	*/
	SparseMsg* sm = new SparseMsg( e2.element(), syn );
	assert( sm );
	const Finfo* f1 = ic->findFinfo( "spike" );
	const Finfo* f2 = Synapse::initCinfo()->findFinfo( "addSpike" );
	assert( f1 && f2 );
	f1->addMsg( f2, sm->mid(), t2 );
	sm->randomConnect( connectionProbability );
	sm->loadBalance( numThreads );

	unsigned int nd = syn->dataHandler()->numData();
//	cout << "Num Syn = " << nd << endl;
	assert( nd == NUMSYN );
	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	bool ret = Field< double >::setVec( e2, "Vm", temp );
	assert( ret );

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


	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );

	/*
	*/
	SingleMsg* m = new SingleMsg( er0, e2 );
	const Finfo* p1 = Tick::initCinfo()->findFinfo( "process0" );
	const Finfo* p2 = ic->findFinfo( "process" );
	ret = p1->addMsg( p2, m->mid(), ticke );

	// ret = SingleMsg::add( er0, "process0", e2, "process" );
	assert( ret );

	// printGrid( i2(), "Vm", 0, thresh );
	Element* se = Id()();
	Shell* s = reinterpret_cast< Shell* >( se->dataHandler()->data( 0 ) );
	s->setclock( 0, timestep, 0 );

	IntFire* ifire100 = reinterpret_cast< IntFire* >( e2.element()->dataHandler()->data( 100 ) );
	IntFire* ifire900 = reinterpret_cast< IntFire* >( e2.element()->dataHandler()->data( 900 ) );

	// Does reinit too.
	s->start( static_cast< double >( timestep * runsteps) + 0.1 );
	assert( fabs( ifire100->getVm() - Vm100 ) < 1e-6 );
	assert( fabs( ifire900->getVm() - Vm900 ) < 1e-6 );

	// cout << "Done ThreadIntFireNetwork" << flush;
	cout << "." << flush;
	// delete i2();
	synId.destroy();
	i2.destroy();
	// synId.destroy();
}

void testMultiNodeIntFireNetwork()
{
	// Known value from single-thread run, at t = 1 sec.
	static const double Vm100 = 0.0857292;
	static const double Vm900 = 0.107449;
	// static const double Vm100 = 0.10124059893763067;
	// static const double Vm900 = 0.091409481280996352;
	// static const unsigned int NUMSYN = 104576;
	static const double thresh = 0.2;
	static const double Vmax = 1.0;
	static const double refractoryPeriod = 0.4;
	static const double weightMax = 0.02;
	static const double delayMax = 4;
	static const double timestep = 0.2;
	static const double connectionProbability = 0.1;
	static const unsigned int runsteps = 5;
	// These are the starting indices of synapses on
	// IntFire[0], [100], [200], ...
	static unsigned int synIndices[] = {
		0, 10355, 20696, 30782, 41080,
		51226, 61456, 71579, 81765, 92060,
		102178,
	};
	// static const unsigned int runsteps = 1000;
	// const Cinfo* ic = IntFire::initCinfo();
	// const Cinfo* sc = Synapse::initCinfo();
	unsigned int size = 1024;
	string arg;
	Eref sheller( Id().eref() );
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	// Qinfo::mergeQ( 0 );

	//mtseed( 5489UL ); // The default value, but better to be explicit.

	vector< unsigned int > dims( 1, size );
	Id i2 = shell->doCreate( "IntFire", Id(), "test2", dims );
	assert( i2()->name() == "test2" );
	Eref e2 = i2.eref();

	// FieldElement< Synapse, IntFire, &IntFire::synapse > syn( sc, i2(), &IntFire::getNumSynapses, &IntFire::setNumSynapses );
	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert( syn->name() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( syn, di );

	unsigned int numThreads = 1;
	if ( Qinfo::numSimGroup() >= 2 ) {
		numThreads = Qinfo::simGroup( 1 )->numThreads;
	}

	MsgId mid = shell->doAddMsg( "Sparse", e2.fullId(), "spike",
		FullId( synId, 0 ), "addSpike" );
	
	const Msg* m = Msg::getMsg( mid );
	Eref mer = m->manager( m->id() );

	SetGet2< double, long >::set( mer, "setRandomConnectivity", 
		connectionProbability, 5489UL );

	SetGet1< unsigned int >::set( mer, "loadBalance", numThreads ); 
	vector< unsigned int > synArraySizes;
	unsigned int start = syn->dataHandler()->getNumData2( synArraySizes );
	// cout << "start = " << start << endl;
	unsigned int synIndex = start;
	for ( unsigned int i = 0; i < size; ++i ) {
		// if ( ( i % 100 ) == 0 ) cout << "i = " << i << "SynIndex = " << synIndex << endl;
		synIndex += synArraySizes[i];
	}

	unsigned int nd = syn->dataHandler()->numData();
	// cout << "Num Syn = " << nd << endl;
	nd = 104576;

	// This fails on multinodes.
	// assert( nd == NUMSYN );
	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	double origVm100 = temp[100];
	double origVm900 = temp[900];

	bool ret = Field< double >::setVec( e2, "Vm", temp );
	assert( ret );

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
	for ( unsigned int i = 0; i < nd; ++i ) {
		weight.push_back( mtrand() * weightMax );
		delay.push_back( mtrand() * delayMax );
	}
	ret = Field< double >::setVec( syne, "weight", weight );
	assert( ret );
	ret = Field< double >::setVec( syne, "delay", delay );
	assert( ret );

	for ( unsigned int i = 0; i < size; i+= 100 ) {
		double wt = Field< double >::get( 
			Eref( syne.element(), DataId( i, 0 ) ), "weight" );
		assert( fabs( wt - weight[ synIndices[ i / 100 ] ] ) < 1e-6 );
		// cout << "Actual wt = " << wt << ", expected = " << weight[ synIndices[ i / 100 ] ] << endl;
	}

	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );

	shell->doAddMsg( "Single", er0.fullId(), "process0",
		e2.fullId(), "process" );
	shell->setclock( 0, timestep, 0 );

	double retVm100 = Field< double >::get( Eref( e2.element(), 100 ), "Vm" );
	double retVm900 = Field< double >::get( Eref( e2.element(), 900 ), "Vm" );
	assert( fabs( retVm100 - origVm100 ) < 1e-6 );
	assert( fabs( retVm900 - origVm900 ) < 1e-6 );

	shell->doStart( static_cast< double >( timestep * runsteps) + 0.0 );
	retVm100 = Field< double >::get( Eref( e2.element(), 100 ), "Vm" );
	retVm900 = Field< double >::get( Eref( e2.element(), 900 ), "Vm" );

	// cout << "MultiNodeIntFireNetwork: Vm100 = " << retVm100 << ", " << Vm100 << "; Vm900 = " << retVm900 << ", " << Vm900 << endl;
	assert( fabs( retVm100 - Vm100 ) < 1e-6 );
	assert( fabs( retVm900 - Vm900 ) < 1e-6 );

	cout << "." << flush;
	shell->doDelete( synId );
	shell->doDelete( i2 );
}
	
void speedTestMultiNodeIntFireNetwork()
{
	static const double thresh = 0.2;
	static const double Vmax = 1.0;
	static const double refractoryPeriod = 0.4;
	static const double weightMax = 0.02;
	static const double delayMax = 4;
	static const double timestep = 0.2;
	static const double connectionProbability = 0.1;
	static const unsigned int runsteps = 1000;
	unsigned int size = 1024;
	string arg;
	Eref sheller( Id().eref() );
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );

	vector< unsigned int > dims( 1, size );
	Id i2 = shell->doCreate( "IntFire", Id(), "test2", dims );
	assert( i2()->name() == "test2" );
	Eref e2 = i2.eref();

	Id synId( i2.value() + 1 );
	Element* syn = synId();
	assert( syn->name() == "synapse" );

	assert( syn->dataHandler()->numData() == 0 );

	DataId di( 1, 0 ); // DataId( data, field )
	Eref syne( syn, di );

	unsigned int numThreads = 1;
	if ( Qinfo::numSimGroup() >= 2 ) {
		numThreads = Qinfo::simGroup( 1 )->numThreads;
	}

	MsgId mid = shell->doAddMsg( "Sparse", e2.fullId(), "spike",
		FullId( synId, 0 ), "addSpike" );
	
	const Msg* m = Msg::getMsg( mid );
	Eref mer = m->manager( m->id() );

	SetGet2< double, long >::set( mer, "setRandomConnectivity", 
		connectionProbability, 5489UL );

	SetGet1< unsigned int >::set( mer, "loadBalance", numThreads ); 
	vector< unsigned int > synArraySizes;
	unsigned int start = syn->dataHandler()->getNumData2( synArraySizes );
	unsigned int synIndex = start;

	unsigned int nd = syn->dataHandler()->numData();
	nd = 104576;

	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	double origVm100 = temp[100];
	double origVm900 = temp[900];

	bool ret = Field< double >::setVec( e2, "Vm", temp );
	assert( ret );

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
	for ( unsigned int i = 0; i < nd; ++i ) {
		weight.push_back( mtrand() * weightMax );
		delay.push_back( mtrand() * delayMax );
	}
	ret = Field< double >::setVec( syne, "weight", weight );
	assert( ret );
	ret = Field< double >::setVec( syne, "delay", delay );
	assert( ret );

	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );

	shell->doAddMsg( "Single", er0.fullId(), "process0",
		e2.fullId(), "process" );
	shell->setclock( 0, timestep, 0 );

	shell->doStart( static_cast< double >( timestep * runsteps) + 0.0 );

	cout << "." << flush;
	shell->doDelete( synId );
	shell->doDelete( i2 );
	shell->doQuit();
}

void testScheduling()
{
	setupTicks();
	testThreads();
	testThreadIntFireNetwork();
	testMultiNodeIntFireNetwork();
}

void testMpiScheduling()
{
	testMultiNodeIntFireNetwork();
}
