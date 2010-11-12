/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/*
#include <string>
using namespace std;
*/

#include <pthread.h>
#include "header.h"
#include "Tick.h"
#include "TickMgr.h"

double TickMgr::EPSILON = 1.0e-9;

static bool tickPtrCmp ( const Tick* i, const Tick* j) 
{ 
	return ( *i < *j );
}

TickMgr::TickMgr()
	: dt_( 1.0 ), nextTime_( 1.0 )
	// assumes zero size of ticks_ vector
{;}
		
TickMgr::TickMgr( Tick* ptr )
	: dt_( ptr->getDt() ), nextTime_( ptr->getDt() )
{
	ticks_.push_back( ptr );
}

/**
* Add a tick to the specified TickMgr, provided dt is OK.
* The new tick is positioned right away, according to stage.
* This is inefficient, but we don't expect to have many ticks,
* typically under 10.
* Cannot use this if a run is already in progress: will need to do
* something with the ProcInfo if we need to.
* Returns true if the dt matches and the add was successful.
*/
bool TickMgr::addTick( const Tick* t )
{
	if ( t->getDt() < EPSILON )
		return 0;
	if ( ticks_.size() == 0 ) {
		ticks_.push_back( t );
		nextTime_ = dt_ = t->getDt();
		tickerator_ = ticks_.begin();
		return 1;
	}

	// if ( fabs( t->getDt() - dt_ ) < EPSILON )
	if ( doubleEq( t->getDt(), dt_ ) ) 
	{
		ticks_.push_back( t );
		sort( ticks_.begin(), ticks_.end(), tickPtrCmp );
		tickerator_ = ticks_.begin();
		return 1;
	}
	return 0;
}

/**
 * Advance the simulation till the specified end time, without
 * worrying about other dts.
 * The Eref e has to refer to the Tick, not the clock.
 * Slightly modified to use a local variable to make it thread-friendly.
 */
/*
void TickMgr::advance( Element* e, ProcInfo* p, double endTime ) {
	while ( nextTime_ < endTime ) {
		p->currTime = nextTime_;
		for ( vector< const Tick* >::iterator i = ticks_.begin(); 
			i != ticks_.end(); ++i )
		{
			(*i)->advance( e, p );
		}
		nextTime_ += dt_;
	}
}
*/

// procInfo is independent for each thread, need to ensure it is updated
// before doing 'advance'.
void TickMgr::advance( Element* e, ProcInfo* p, double endTime ) 
{
	p->dt = dt_;
	double nt = nextTime_; // use an independent timer for each thread.
	// cout << "TickMgr::advance: nextTime_ = " << nextTime_ << ", endTime = " << endTime << ", thread = " << p->threadId << endl;
	while ( nt < endTime ) {
		p->currTime = nt;
		nt += dt_;

		for ( vector< const Tick* >::iterator i = ticks_.begin(); 
			i != ticks_.end(); ++i )
		{
			(*i)->advance( e, p ); // This calls barrier just before clearQ.
		}
	}

	if ( p->threadId == 0 ) {
		nextTime_ = nt;
	}
	if ( p->barrier1 ) {
		int rc = pthread_barrier_wait( 
			reinterpret_cast< pthread_barrier_t* >( p->barrier1 ) );
		assert( rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD );
	}
	/*
	*/
}

double TickMgr::getNextTime() const
{
	return nextTime_;
}

double TickMgr::getDt() const
{
	return dt_;
}

void TickMgr::reinit( const Eref& e, ProcInfo* p )
{
	nextTime_ = dt_;
	for ( vector< const Tick* >::iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i )
	{
		(*i)->reinit( e, p );
	}
}


///////////////////////////////////////////////////////////////////////
// New version here.
///////////////////////////////////////////////////////////////////////

void TickMgr::advancePhase1( ProcInfo* p ) const
{
	p->dt = dt_;
	p->currTime = nextTime_;
	assert( ticks_.size() > 0 );
	( *tickerator_ )->advance( p ); // Move one tick along.
}

// This modifies the TickMgr, has to happen on a single thread and
// safely isolated from the multithread ops in advancePhase1.
void TickMgr::advancePhase2( ProcInfo* p ) 
{
	++tickerator_;
	if ( tickerator_ == ticks_.end() ) {
		tickerator_ = ticks_.begin(); 
		nextTime_ += dt_;
	}
}

void TickMgr::reinitPhase1( ProcInfo* p ) const
{
	for ( vector< const Tick* >::const_iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i )
	{
		(*i)->reinit( p );
	}
}

void TickMgr::reinitPhase2( ProcInfo* p )
{
	nextTime_ = dt_;
	tickerator_ = ticks_.begin();
}
