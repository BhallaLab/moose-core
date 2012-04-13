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
	: dt_( 1.0 ), nextTime_( 1.0 ), tickerator_( 0 )
	// assumes zero size of ticks_ vector
{;}
		
TickMgr::TickMgr( Tick* ptr )
	: dt_( ptr->getDt() ), nextTime_( ptr->getDt() ), tickerator_( 0 )
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
		tickerator_ = 0;
		return 1;
	}

	// if ( fabs( t->getDt() - dt_ ) < EPSILON )
	if ( doubleEq( t->getDt(), dt_ ) ) 
	{
		ticks_.push_back( t );
		sort( ticks_.begin(), ticks_.end(), tickPtrCmp );
		tickerator_ = 0;
		return 1;
	}
	return 0;
}


double TickMgr::getNextTime() const
{
	return nextTime_;
}

void TickMgr::setNextTime( double t )
{
	if ( t >= 0 )
		nextTime_ = t;
}

double TickMgr::getDt() const
{
	return dt_;
}

/*
void TickMgr::reinit( const Eref& e, ProcInfo* p )
{
	nextTime_ = dt_;
	for ( vector< const Tick* >::iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i )
	{
		(*i)->reinit( e, p );
	}
}
*/

bool TickMgr::isInited() const 
{
	return ( tickerator_ < ticks_.size() );
}

///////////////////////////////////////////////////////////////////////
// New version here.
///////////////////////////////////////////////////////////////////////

void TickMgr::advancePhase1( ProcInfo* p ) const
{
	p->dt = dt_;
	p->currTime = nextTime_;
	assert( ticks_.size() > tickerator_ );
	ticks_[ tickerator_ ]->advance( p ); // move one tick along.
	// ( *tickerator_ )->advance( p ); // Move one tick along.
}

// This modifies the TickMgr, has to happen on a single thread and
// safely isolated from the multithread ops in advancePhase1.
void TickMgr::advancePhase2( ProcInfo* p ) 
{
	++tickerator_;
	if ( tickerator_ == ticks_.size() ) {
		tickerator_ = 0;
		nextTime_ += dt_;
	}
}

void TickMgr::reinitPhase0()
{
	tickerator_ = 0;
	nextTime_ = dt_;
}

void TickMgr::reinitPhase1( ProcInfo* p ) const
{
	p->dt = dt_;
	p->currTime = 0.0;
	assert( ticks_.size() > tickerator_ );
	ticks_[ tickerator_ ]->reinit( p );
/*

	for ( vector< const Tick* >::const_iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i )
	{
		(*i)->reinit( p );
	}
	*/
}

bool TickMgr::reinitPhase2( ProcInfo* p )
{
	++tickerator_;
	if ( tickerator_ == ticks_.size() ) {
		tickerator_ = 0;
		nextTime_ = dt_;
		return 1;
	}
	return 0;
}

const vector< const Tick* >& TickMgr::ticks() const
{
	return ticks_;
}
