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

/**
 * The Tick handles the nuts and bolts of scheduling. It sends the
 * Process (or other event) message to all the scheduled objects,
 * and it keeps track of the update sequence with its sibling Tick
 * objects.
 */

/**
 * We reserve a large number of slots, since each Tick uses a separate one.
 * Should ideally find a way to predefine this to the max number of Ticks.
 */
const ConnId procSlot = 10;

static SrcFinfo1< ProcPtr >* process = 
	new SrcFinfo1< ProcPtr >(
		"process",
		"Calls Process on target Elements. May need special option.",
		procSlot
);


const Cinfo* Tick::initCinfo()
{

	static Finfo* tickFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		// Refers it to the parent Clock.
		new UpValueFinfo< Clock, double >(
			"dt",
			"Timestep for this tick",
			&Clock::setTickDt,
			&Clock::getTickDt
		),
		new ValueFinfo< Tick, double >(
			"localdt",
			"Timestep for this tick",
			&Tick::setDt,
			&Tick::getDt
		),
		new UpValueFinfo< Clock, unsigned int >(
			"stage",
			"Sequence number if multiple ticks have the same dt.",
			&Clock::setStage,
			&Clock::getStage
		),
		new ValueFinfo< Tick, string>(
			"path",
			"Wildcard path of objects managed by this tick",
			&Tick::setPath,
			&Tick::getPath
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		process,
		// clearQ,

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	};
	
	static Cinfo tickCinfo(
		"Tick",
		0,
		tickFinfos,
		sizeof(tickFinfos) / sizeof(Finfo *),
		new FieldDinfo()
	);

	return &tickCinfo;
}

static const Cinfo* tickCinfo = Tick::initCinfo();

///////////////////////////////////////////////////
// Tick class definition functions
///////////////////////////////////////////////////
Tick::Tick()
	: dt_( 0.0 ), stage_( 0 )
{ ; }

Tick::~Tick()
{ ; }

bool Tick::operator<( const Tick& other ) const
{
	const double EPSILON = 1e-9;

	if ( dt_ < other.dt_ ) return 1;
		if ( fabs( 1.0 - dt_ / other.dt_ ) < EPSILON && 
			stage_ < other.stage_ )
			return 1;
	return 0;
}


bool Tick::operator==( const Tick& other ) const
{
	return ( dt_ == other.dt_ && stage_ == other.stage_ );
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

/**
 * This is called when dt is set on the local Tick.
 * We first fix up local values for nextTime and dt, then
 * we ask the parent ClockJob to re-sort the clock ticks to
 * put them back in order.
 * There should also be a bit here to set the parent.
 */
void Tick::setDt( double newdt )
{
	/*
	nextTime_ += newdt - dt_;
	if ( nextTime_ < dt_ )
		nextTime_ = dt_;
	*/
	dt_ = newdt;
}
/**
 * The getDt just looks up the local dt, much less involved than
 * the setDt function.
 */
double Tick::getDt() const
{
	return dt_;
}

/**
 * This is called when stage is set on the local Tick.
 * Like the setDt, it has to ask the parent ClockJob to
 * re-sort the clock ticks to put them back in order.
 */
void Tick::setStage( unsigned int v )
{
	stage_ = v;
}

/**
 * The getStage just looks up the local stage, much less involved than
 * the setStage function.
 */
unsigned int Tick::getStage() const
{
	return stage_;
}

/**
 * nextTime is here to peek into when the tick is due to fire next.
 * Not clear if it should become private.
double Tick::getNextTime() const
{
	// return nextTime_;
	return 0.0;
}
 */

/**
 * set and get Path are problematic. Their goal is to assign the 
 * targets for this Tick. As framed, they fit in with the older
 * GENESIS syntax. For now, put in dummy functions.
 */
void Tick::setPath( string v )
{
	path_ = v;
	// Something here to change the messages
}

string Tick::getPath() const
{
	return path_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Virtual function definitions for actually sending out the 
// process and reinit calls.
///////////////////////////////////////////////////
/**
 * This sends out the process call.
 */
void Tick::advance( Element* e, ProcInfo* info ) const
{
	cout << "(" << dt_ << ", " << stage_ << " ) at " << info->currTime << endl;
	// Hack: we need a better way to define which connId to use.
	// Presumably we should at least take an offset from the predefined
	// Slots like children.
	const Conn* c = e->conn( index_ );
	c->clearQ();
	c->process( info );
}

void Tick::setIndex( unsigned int index ) 
{
	index_ = index;
}

/**
 * This sends out the call to reinit objects. It is virtualized
 * because derived classes (ParTick) need to do much more complicated
 * things to coordinate the reinit.
 */
void Tick::reinit( Eref e ) const
{
	;
	// nextTime_ = dt_;
	// send1< ProcInfo >( e, reinitSlot, info );
}
