/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Message.h"
#include "Tick.h"
#include "TickPtr.h"
#include "Clock.h"
#include "testScheduling.h"

#include <queue>
#include "../biophysics/Synapse.h"
#include "../biophysics/IntFire.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "PsparseMsg.h"
#include "../randnum/randnum.h"

/**
 * Check that the ticks are set up properly, created and destroyed as
 * needed, and are sorted when dts are assigned
 */
void setupTicks()
{
	static const double EPSILON = 1.0e-9;
	// const Cinfo* tc = Tick::initCinfo();
	Id clock = Id::nextId();
	vector< unsigned int > dims( 1, 1 );
	Element* clocke = new Element( clock, Clock::initCinfo(), "tclock", dims );
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

	bool ret = OneToAllMsg::add( clocker, "childTick", ticke, "parent" );
	assert( ret );

	assert( ticke->dataHandler()->numData() == 0 );
	ret = Field< unsigned int >::set( clocker, "numTicks", size );
	assert( ret );
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


	testSchedElement tse;
	Eref ts( &tse, 0 );
	
	// No idea what FuncId to use here. Assume 0.
	FuncId f( 0 );
	SingleMsg m0( er0, ts ); er0.element()->addMsgAndFunc( m0.mid(), f, 0 );
	SingleMsg m1( er1, ts ); er1.element()->addMsgAndFunc( m1.mid(), f, 1 );
	SingleMsg m2( er2, ts ); er2.element()->addMsgAndFunc( m2.mid(), f, 2 );
	SingleMsg m3( er3, ts ); er3.element()->addMsgAndFunc( m3.mid(), f, 3 );
	SingleMsg m4( er4, ts ); er4.element()->addMsgAndFunc( m4.mid(), f, 4 );
	SingleMsg m5( er5, ts ); er5.element()->addMsgAndFunc( m5.mid(), f, 7 );

	Qinfo q( 0, 0, 8 );
	cdata->start( clocker, &q, 20 );

	cout << "." << flush;

	delete clocke;
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

	testThreadSchedElement tse;
	Eref ts( &tse, 0 );
	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );
	Eref er1( ticke, DataId( 0, 1 ) );
	Eref er2( ticke, DataId( 0, 2 ) );
	Eref er3( ticke, DataId( 0, 3 ) );
	Eref er4( ticke, DataId( 0, 4 ) );
	Eref er5( ticke, DataId( 0, 5 ) );

	// No idea what FuncId to use here. Assume 0.
	FuncId f( 0 );
	SingleMsg m0( er0, ts ); er0.element()->addMsgAndFunc( m0.mid(), f, 0 );
	SingleMsg m1( er1, ts ); er1.element()->addMsgAndFunc( m1.mid(), f, 1 );
	SingleMsg m2( er2, ts ); er2.element()->addMsgAndFunc( m2.mid(), f, 2 );
	SingleMsg m3( er3, ts ); er3.element()->addMsgAndFunc( m3.mid(), f, 3 );
	SingleMsg m4( er4, ts ); er4.element()->addMsgAndFunc( m4.mid(), f, 4 );
	SingleMsg m5( er5, ts ); er5.element()->addMsgAndFunc( m5.mid(), f, 5 );
	s->start( 10 );

	cout << "Done TestThreads" << flush;
	cout << "." << flush;
}

void testThreadIntFireNetwork()
{
	// Known value from single-thread run, at t = 1 sec.
	static const double Vm100 = 0.0857292;
	static const double Vm900 = 0.107449;
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
	bool ret = PsparseMsg::add( e2.element(), "spike", syn, "addSpike", 
		connectionProbability, numThreads ); // Include group id as an arg. 
	assert( ret );

	unsigned int nd = syn->dataHandler()->numData();
//	cout << "Num Syn = " << nd << endl;
	assert( nd == NUMSYN );
	vector< double > temp( size, 0.0 );
	for ( unsigned int i = 0; i < size; ++i )
		temp[i] = mtrand() * Vmax;

	ret = Field< double >::setVec( e2, "Vm", temp );
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

	// printGrid( i2(), "Vm", 0, thresh );
	Element* se = Id()();
	Shell* s = reinterpret_cast< Shell* >( se->dataHandler()->data( 0 ) );
	s->setclock( 0, timestep, 0 );

	Element* ticke = Id( 2 )();
	Eref er0( ticke, DataId( 0, 0 ) );

	ret = SingleMsg::add( er0, "process0", e2, "process" );
	assert( ret );

	IntFire* ifire100 = reinterpret_cast< IntFire* >( e2.element()->dataHandler()->data( 100 ) );
	IntFire* ifire900 = reinterpret_cast< IntFire* >( e2.element()->dataHandler()->data( 900 ) );

	s->start( timestep * runsteps );
	assert( fabs( ifire100->getVm() - Vm100 ) < 1e-6 );
	assert( fabs( ifire900->getVm() - Vm900 ) < 1e-6 );

	cout << "Done ThreadIntFireNetwork" << flush;
	cout << "." << flush;
	delete i2();
}
	

void testScheduling( bool useMPI )
{
	setupTicks();
	testThreads();
	if ( !useMPI )
		testThreadIntFireNetwork();
}
