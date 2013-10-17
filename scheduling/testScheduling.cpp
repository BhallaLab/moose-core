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
#include "TickMgr.h"
#include "TickPtr.h"
#include "Clock.h"
#include "testScheduling.h"

#include <queue>
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "SingleMsg.h"
#include "../builtins/Arith.h"
#include "../randnum/randnum.h"
#include "../shell/Shell.h"

bool TestSched::isInitPending_( 1 );
int TestSched::globalIndex_( 0 );

/**
 * Note that ticks cannot be made independently; they are sub-elements
 * of the Clock.
 */
void testTicks()
{
	Eref sheller( Id().eref() );
	Shell* shell = reinterpret_cast< Shell* >( sheller.data() );
	vector< int > dims( 1, 1 );
	Id clockId = shell->doCreate( "Clock", Id(), "tclock", dims );
	Id tickId = Id( clockId.value() + 1 );
	Id arithId = shell->doCreate( "Arith", clockId, "arith", dims );
	ProcInfo p;

	MsgId m1 = shell->doAddMsg( "Single", 
		ObjId( tickId, 0 ), "proc0", ObjId( arithId, 0 ), "proc" );
	assert( m1 != Msg::bad );

	vector< Id > msgDests;
	const Finfo* sf = Tick::initCinfo()->findFinfo( "process0" );
	assert( sf );
	tickId()->getNeighbours( msgDests, sf );
	assert( msgDests.size() == 1 );
	assert( msgDests[0] == arithId );

	bool ret = Field< double >::set( tickId, "dt", 5.0);
	assert( ret );

	Tick* t0 = reinterpret_cast< Tick* >( tickId.eref().data() );
	assert( doubleEq( 5.0, t0->dt_ ) );

	// By default it gets hard-code set to the systiem tick Element, so
	// we need to reassign it here for this test.
	t0->setElement( tickId() ); 

	Arith* a0 = reinterpret_cast< Arith* >( arithId.eref().data() );
	a0->arg1_ = 123.4;
	a0->arg2_ = 7;
	a0->arg3_ = 9;

	t0->reinit( &p );
	assert( doubleEq( 0, a0->arg1_ ) );
	assert( doubleEq( 0, a0->arg2_ ) );
	assert( doubleEq( 0, a0->arg3_ ) );

	a0->arg1_ = 123.4;
	a0->arg2_ = 7;
	a0->arg3_ = 9;

	t0->advance( &p );
	assert( doubleEq( 123.4, a0->arg1_ ) );
	assert( doubleEq( 7, a0->arg2_ ) );
	assert( doubleEq( 0, a0->arg3_ ) );
	assert( doubleEq( 139.4, a0->output_ ) );

	const Msg* m = Msg::getMsg( m1 );
	assert( m != 0 );
	assert( tickId() != 0 );
	assert( clockId() != 0 );
	clockId.destroy();
	assert( tickId() != 0 );
	tickId.destroy();
	arithId.destroy();
	assert( clockId() == 0 );
	assert( tickId() == 0 );
	m = Msg::getMsg( m1 );
	assert( m == 0 );

	cout << "." << flush;
}


//////////////////////////////////////////////////////////////////////
// Setting up a class for testing scheduling.
//////////////////////////////////////////////////////////////////////

static DestFinfo processFinfo( "process",
	"handles process call",
	new ProcOpFunc< TestSched >( &TestSched::process ) );
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

void TestSched::process( const Eref& e, ProcPtr p )
{
	static const int timings[] = { 1, 2, 2, 2, 3, 3, 4, 4, 4, 
		5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10,
		11, 12, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15,
		16, 16, 16, 17, 18, 18, 18, 18, 19, 20, 20, 20, 20, 20,
		21, 21, 22, 22, 22, 23, 24, 24, 24, 24, 25, 25, 25 };
	// unsigned int max = sizeof( timings ) / sizeof( int );
	// cout << Shell::myNode() << " : timing[ " << index_ << "  ] = " << timings[ index_ ] << ", time = " << p->currTime << endl;
	if ( static_cast< int >( p->currTime ) != timings[ index_ ] ) {
		cout << Shell::myNode() << ":testSchedElement::process: index= " << index_ << ", currTime = " << p->currTime << endl;
	}

	assert( static_cast< int >( p->currTime ) == timings[ index_ ] );
	++index_;

	// Check that everything remains in sync
	assert( ( globalIndex_ - index_ )*( globalIndex_ - index_ ) <= 1 );
	globalIndex_ = index_;
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
	vector< DimInfo > dims;
	/*
	Id clock = Id::nextId();
	Element* clocke = new Element( clock, Clock::initCinfo(), "tclock",
		dims, 1, true );
		*/
	Id clock(1);
	Element* clocke = clock.element();

	assert( clocke );
	// bool ret = Clock::initCinfo()->create( clock, "tclock", 1 );
	// assert( ret );
	// Element* clocke = clock();
	Eref clocker = clock.eref();
	Id tickId( clock.value() + 1 );
	Element* ticke = tickId();
	assert( ticke->getName() == "tick" );


	/*
	unsigned int size = 10;
	for ( unsigned int i = 0; i < size; ++i ) {
		Eref er( ticke, DataId( i ) );
		reinterpret_cast< Tick* >( er.data() )->setElement( ticke );
	}
	*/

	// cout << Shell::myNode() << ": numTicks: " << ticke->dataHandler()->totalEntries() << ", " << size << endl;
	assert( ticke->dataHandler()->localEntries() == Tick::maxTicks );

	// Here I'm setting Tick fields, but the call is routed through the
	// Clock
	ObjId er0( tickId, DataId( 2 ) );
	bool ret = Field< double >::set( er0, "dt", 5.0);
	assert( ret );
	ObjId er1( tickId, DataId( 1 ) );
	ret = Field< double >::set( er1, "dt", 2.0);
	assert( ret );
	ObjId er2( tickId, DataId( 0 ) );
	ret = Field< double >::set( er2, "dt", 2.0);
	assert( ret );
	ObjId er3( tickId, DataId( 3 ) );
	ret = Field< double >::set( er3, "dt", 1.0);
	assert( ret );
	ObjId er4( tickId, DataId( 4 ) );
	ret = Field< double >::set( er4, "dt", 3.0);
	assert( ret );
	// Note that here I put the tick on a different DataId. later it gets
	// to sit on the appropriate Conn, when the SingleMsg is set up.
	ObjId er5( tickId, DataId( 7 ) );
	ret = Field< double >::set( er5, "dt", 5.0);
	assert( ret );



	Id tsid = Id::nextId();
	Element* tse = new Element( tsid, testSchedCinfo, "tse", dims, 1 );

	Eref ts( tse, 0 );
	
	FuncId f( processFinfo.getFid() );
	const Finfo* proc0 = ticke->cinfo()->findFinfo( "process0" );
	assert( proc0 );
	const SrcFinfo* sproc0 = dynamic_cast< const SrcFinfo* >( proc0 );
	assert( sproc0 );
	unsigned int b0 = sproc0->getBindIndex();
	SingleMsg *m0 = new SingleMsg( Msg::nextMsgId(), er0.eref(), ts ); 
	er0.element()->addMsgAndFunc( m0->mid(), f, er0.dataId.value()*2 + b0);
	SingleMsg *m1 = new SingleMsg( Msg::nextMsgId(), er1.eref(), ts ); 
	er1.element()->addMsgAndFunc( m1->mid(), f, er1.dataId.value()*2 + b0);
	SingleMsg *m2 = new SingleMsg( Msg::nextMsgId(), er2.eref(), ts );
	er2.element()->addMsgAndFunc( m2->mid(), f, er2.dataId.value()*2 + b0);
	SingleMsg *m3 = new SingleMsg( Msg::nextMsgId(), er3.eref(), ts ); 
	er3.element()->addMsgAndFunc( m3->mid(), f, er3.dataId.value()*2 + b0);
	SingleMsg *m4 = new SingleMsg( Msg::nextMsgId(), er4.eref(), ts ); 
	er4.element()->addMsgAndFunc( m4->mid(), f, er4.dataId.value()*2 + b0);
	SingleMsg *m5 = new SingleMsg( Msg::nextMsgId(), er5.eref(), ts ); 
	er5.element()->addMsgAndFunc( m5->mid(), f, er5.dataId.value()*2 + b0);



	Clock* cdata = reinterpret_cast< Clock* >( clocker.data() );
	cdata->rebuild();
	assert( cdata->tickPtr_.size() == 4 );
	assert( fabs( cdata->tickPtr_[0].mgr()->dt_ - 1.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[1].mgr()->dt_ - 2.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[2].mgr()->dt_ - 3.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[3].mgr()->dt_ - 5.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[0].mgr()->nextTime_ - 1.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[1].mgr()->nextTime_ - 2.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[2].mgr()->nextTime_ - 3.0 ) < EPSILON );
	assert( fabs( cdata->tickPtr_[3].mgr()->nextTime_ - 5.0 ) < EPSILON );
	assert( cdata->tickPtr_[0].mgr()->ticks_.size() == 1 );
	assert( cdata->tickPtr_[1].mgr()->ticks_.size() == 2 );
	assert( cdata->tickPtr_[2].mgr()->ticks_.size() == 1 );
	assert( cdata->tickPtr_[3].mgr()->ticks_.size() == 2 );

	assert( cdata->tickPtr_[0].mgr()->ticks_[0] == reinterpret_cast< const Tick* >( er3.data() ) );
	assert( cdata->tickPtr_[1].mgr()->ticks_[0] == reinterpret_cast< const Tick* >( er2.data() ) );
	assert( cdata->tickPtr_[1].mgr()->ticks_[1] == reinterpret_cast< const Tick* >( er1.data() ) );
	assert( cdata->tickPtr_[2].mgr()->ticks_[0] == reinterpret_cast< const Tick* >( er4.data() ) );
	assert( cdata->tickPtr_[3].mgr()->ticks_[0] == reinterpret_cast< const Tick* >( er0.data() ) );
	assert( cdata->tickPtr_[3].mgr()->ticks_[1] == reinterpret_cast< const Tick* >( er5.data() ) );


	Qinfo::emptyAllQs();
	ProcInfo p;
	cdata->handleReinit();
	/*
	 * Now the handleReinit completes the entire reinit as soon as it is
	 * called. The below sequence assumes that it waits for the process
	 * loop.
	assert( cdata->currTickPtr_ == 0 );
	assert( Clock::procState_ == Clock::TurnOnReinit ); 
	cdata->reinitPhase1( &p );
	assert( Clock::procState_ == Clock::TurnOnReinit ); 
	cdata->reinitPhase2( &p );
	assert( Clock::procState_ == Clock::TurnOnReinit ); 
	assert( cdata->tickPtr_.size() == 4 );
	assert( cdata->currTickPtr_ == 1 );
	// cycle 0 done. There are 4 tickPtrs.
	cdata->reinitPhase1( &p );
	cdata->reinitPhase2( &p );
	assert( cdata->currTickPtr_ == 1 ); // We have two ticks for dt = 2.
	cdata->reinitPhase1( &p );
	cdata->reinitPhase2( &p );
	assert( cdata->currTickPtr_ == 2 );
	// cycle 1 done
	cdata->reinitPhase1( &p );
	cdata->reinitPhase2( &p );
	assert( cdata->currTickPtr_ == 3 );
	// cycle 2 done
	cdata->reinitPhase1( &p );
	cdata->reinitPhase2( &p );
	assert( cdata->currTickPtr_ == 3 ); // Two ticks for dt = 5.
	cdata->reinitPhase1( &p );
	cdata->reinitPhase2( &p );
	assert( cdata->currTickPtr_ == 4 );
	// cycle 3 done. Now we should be done
	assert( Clock::procState_ == Clock::TurnOffReinit ); 

	Clock::procState_ = Clock::NoChange;
	// assert( cdata->doingReinit_ == 0 );
	*/

	cdata->handleStart( runtime );

	assert( doubleEq( cdata->getCurrentTime(), runtime ) );
	// Get rid of pending events in the queues.
	Qinfo::emptyAllQs();
	tsid.destroy();
	for ( unsigned int i = 0; i < Tick::maxTicks; ++i ) {
		cdata->ticks_[i].setDt( 0.0 );
	}
	cdata->rebuild();
	assert( cdata->tickMgr_.size() == 0 );
	assert( cdata->tickPtr_.size() == 0 );
	cout << "." << flush;
}

/// Tests how the scheduling ticks have been configured.
void testTickConfig()
{
	Id tick( 2 );
	assert( tick.element() );
	assert( tick.element()->getName() == "tick" );
	DataHandler* dh = tick.element()->dataHandler();
	assert( dh );
	FieldDataHandlerBase* fdb = dynamic_cast< FieldDataHandlerBase *>(dh);
	assert( fdb );

	assert ( fdb->localEntries() == Tick::maxTicks );
	assert ( dh->localEntries() == Tick::maxTicks );

	assert( dh->numDimensions() == 1 );
	assert( dh->pathDepth() == 2 );
	assert( dh->sizeOfDim( 0 ) == 16 );
	assert( dh->sizeOfDim( 1 ) == 0 );

	assert( dh->dims().size() == 1 );
	assert( dh->dims()[0].size == 16 );
	assert( dh->dims()[0].depth == 2 );
	assert( dh->dims()[0].isRagged );
	assert( dh->getFieldArraySize( 0 ) == Tick::maxTicks );
	assert( dh->fieldMask() == 0x0f);
	assert( fdb->numFieldBits() == 4 );

	unsigned int i = dh->linearIndex( 123 );
	// The zeroDimHandler clears out any attempt to get indices other than
	// zero, so the parent index terms in the linearIndex vanish.
	unsigned int j = 123 % 16; 
	assert( i == j );

	for ( unsigned int i = 0; i < Tick::maxTicks; ++i ) {
		vector< vector< unsigned int > > pathIndices = 
				dh->pathIndices( i );
		assert( pathIndices.size() == 3 );
		assert( pathIndices[0].size() == 0 );
		assert( pathIndices[1].size() == 0 );
		assert( pathIndices[2].size() == 1 );
		assert( pathIndices[2][0] == i );
	}

	cout << "." << flush;
}

void testScheduling()
{
	testTicks();
	setupTicks();
}

void testSchedulingProcess()
{
	testTickConfig(); // Checks that the system has correctly built ticks
}

void testMpiScheduling()
{
}
