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

#include "header.h"
#include "Tick.h"
#include "TickPtr.h"

double TickPtr::EPSILON = 1.0e-9;

static bool tickPtrCmp ( const Tick* i, const Tick* j) 
{ 
	return ( *i < *j );
}

TickPtr::TickPtr()
	: dt_( 1.0 ), nextTime_( 1.0 ) // assumes zero size of ticks_
{;}
		
TickPtr::TickPtr( Tick* ptr )
	: dt_( ptr->getDt() ), nextTime_( ptr->getDt() )
{
	ticks_.push_back( ptr );
}

bool TickPtr::operator<( const TickPtr other ) const {
	return ( nextTime_ < other.nextTime_ );
}

/**
* Add a tick to the specified TickPtr, provided dt is OK.
* The new tick is positioned right away, according to stage.
* This is inefficient, but we don't expect to have many ticks,
* typically under 10.
* Cannot use this if a run is already in progress: will need to do
* something with the ProcInfo if we need to.
* Returns true if the dt matches and the add was successful.
*/
bool TickPtr::addTick( const Tick* t )
{
	if ( t->getDt() < EPSILON )
		return 0;
	if ( ticks_.size() == 0 ) {
		ticks_.push_back( t );
		nextTime_ = dt_ = t->getDt();
		return 1;
	}
	if ( fabs( t->getDt() - dt_ ) < EPSILON ) {
		ticks_.push_back( t );
		sort( ticks_.begin(), ticks_.end(), tickPtrCmp );
		return 1;
	}
	return 0;
}

/**
 * Advance the simulation till the specified end time, without
 * worrying about other dts.
 * The Eref e has to refer to the Tick, not the clock.
 */
void TickPtr::advance( Element* e, ProcInfo* p, double endTime ) {
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

/*
void TickPtr::advanceThread( Element* e, ProcInfo* p, double endTime ) {
	double nt = nextTime_;
	while ( nt < endTime ) {
		p->currTime = nt;
		for ( vector< const Tick* >::iterator i = ticks_.begin(); 
			i != ticks_.end(); ++i )
		{
			(*i)->advanceThread( e, p );
		}
		nt += dt_;
		if ( p->threadId == FIRSTWORKER ) // first worker thread
			nextTime_ = nt;
	}
}
*/

double TickPtr::getNextTime() const
{
	return nextTime_;
}

void TickPtr::reinit( Eref e )
{
	nextTime_ = dt_;
	for ( vector< const Tick* >::iterator i = ticks_.begin(); 
		i != ticks_.end(); ++i )
	{
		(*i)->reinit( e );
	}
}

		
