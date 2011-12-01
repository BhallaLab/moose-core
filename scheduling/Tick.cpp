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

/**
 * The Tick handles the nuts and bolts of scheduling. It sends the
 * Process (or other event) message to all the scheduled objects,
 * and it keeps track of the update sequence with its sibling Tick
 * objects.
 */

// This vector contains the SrcFinfos used for Process calls for each
// of the Ticks.
// vector< SrcFinfo* > process;

SrcFinfo ** processVec() {
	static SrcFinfo1< ProcPtr > process0( "process0", "Process for Tick 0" );
	static SrcFinfo1< ProcPtr > process1( "process1", "Process for Tick 1" );
	static SrcFinfo1< ProcPtr > process2( "process2", "Process for Tick 2" );
	static SrcFinfo1< ProcPtr > process3( "process3", "Process for Tick 3" );
	static SrcFinfo1< ProcPtr > process4( "process4", "Process for Tick 4" );
	static SrcFinfo1< ProcPtr > process5( "process5", "Process for Tick 5" );
	static SrcFinfo1< ProcPtr > process6( "process6", "Process for Tick 6" );
	static SrcFinfo1< ProcPtr > process7( "process7", "Process for Tick 7" );
	static SrcFinfo1< ProcPtr > process8( "process8", "Process for Tick 8" );
	static SrcFinfo1< ProcPtr > process9( "process9", "Process for Tick 9" );

	static SrcFinfo* processVec[] = {
		&process0, &process1, &process2, &process3, &process4, &process5, &process6, &process7, &process8, &process9, };

	return processVec;
}

SrcFinfo ** reinitVec() {

	static SrcFinfo1< ProcPtr > reinit0( "reinit0", "Reinit for Tick 0" );
	static SrcFinfo1< ProcPtr > reinit1( "reinit1", "Reinit for Tick 1" );
	static SrcFinfo1< ProcPtr > reinit2( "reinit2", "Reinit for Tick 2" );
	static SrcFinfo1< ProcPtr > reinit3( "reinit3", "Reinit for Tick 3" );
	static SrcFinfo1< ProcPtr > reinit4( "reinit4", "Reinit for Tick 4" );
	static SrcFinfo1< ProcPtr > reinit5( "reinit5", "Reinit for Tick 5" );
	static SrcFinfo1< ProcPtr > reinit6( "reinit6", "Reinit for Tick 6" );
	static SrcFinfo1< ProcPtr > reinit7( "reinit7", "Reinit for Tick 7" );
	static SrcFinfo1< ProcPtr > reinit8( "reinit8", "Reinit for Tick 8" );
	static SrcFinfo1< ProcPtr > reinit9( "reinit9", "Reinit for Tick 9" );

	static SrcFinfo* reinitVec[] = {
		&reinit0, &reinit1, &reinit2, &reinit3, &reinit4, &reinit5, &reinit6, &reinit7, &reinit8, &reinit9, };

	return reinitVec;
}

const unsigned int Tick::maxTicks = 10;

const Cinfo* Tick::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		// Refers it to the parent Clock.
		static UpValueFinfo< Clock, double > dt(
			"dt",
			"Timestep for this tick",
			&Clock::setTickDt,
			&Clock::getTickDt
		);
		static ValueFinfo< Tick, double > localdt(
			"localdt",
			"Timestep for this tick",
			&Tick::setDt,
			&Tick::getDt
		);
		/*
		static ValueFinfo< Tick, string> path(
			"path",
			"Wildcard path of objects managed by this tick",
			&Tick::setPath,
			&Tick::getPath
		);
		*/
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// These are defined outside this function.

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		/*
		static DestFinfo parent( "parent", 
			"Message from Parent Element(s)", 
			new EpFunc0< Tick >( &Tick::destroy ) );
			*/

	static Finfo* procShared0[] = { processVec()[0], reinitVec()[0] };
	static Finfo* procShared1[] = { processVec()[1], reinitVec()[1] };
	static Finfo* procShared2[] = { processVec()[2], reinitVec()[2] };
	static Finfo* procShared3[] = { processVec()[3], reinitVec()[3] };
	static Finfo* procShared4[] = { processVec()[4], reinitVec()[4] };
	static Finfo* procShared5[] = { processVec()[5], reinitVec()[5] };
	static Finfo* procShared6[] = { processVec()[6], reinitVec()[6] };
	static Finfo* procShared7[] = { processVec()[7], reinitVec()[7] };
	static Finfo* procShared8[] = { processVec()[8], reinitVec()[8] };
	static Finfo* procShared9[] = { processVec()[9], reinitVec()[9] };

	static SharedFinfo proc0( "proc0", "Shared proc/reinit message", procShared0, sizeof( procShared0 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc1( "proc1", "Shared proc/reinit message", procShared1, sizeof( procShared1 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc2( "proc2", "Shared proc/reinit message", procShared2, sizeof( procShared2 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc3( "proc3", "Shared proc/reinit message", procShared3, sizeof( procShared3 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc4( "proc4", "Shared proc/reinit message", procShared4, sizeof( procShared4 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc5( "proc5", "Shared proc/reinit message", procShared5, sizeof( procShared5 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc6( "proc6", "Shared proc/reinit message", procShared6, sizeof( procShared6 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc7( "proc7", "Shared proc/reinit message", procShared7, sizeof( procShared7 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc8( "proc8", "Shared proc/reinit message", procShared8, sizeof( procShared8 ) / sizeof( const Finfo* ) );
	static SharedFinfo proc9( "proc9", "Shared proc/reinit message", procShared9, sizeof( procShared9 ) / sizeof( const Finfo* ) );

	static Finfo* tickFinfos[] =
	{
		// Fields
		&dt,
		&localdt,
		// &path,
		// Shared SrcFinfos for process
		&proc0,
		&proc1,
		&proc2,
		&proc3,
		&proc4,
		&proc5,
		&proc6,
		&proc7,
		&proc8,
		&proc9,
		// MsgDest definitions
		// &parent, // I thought this was to be inherited?
	};
	
	static Cinfo tickCinfo(
		"Tick",
		// 0,
		Neutral::initCinfo(),
		tickFinfos,
		sizeof(tickFinfos) / sizeof(Finfo *),
		new Dinfo< Tick >()
	);

	return &tickCinfo;
}

static const Cinfo* tickCinfo = Tick::initCinfo();

///////////////////////////////////////////////////
// Tick class definition functions
///////////////////////////////////////////////////
Tick::Tick()
	: dt_( 0.0 ), index_( 0 )
{ ; }

Tick::~Tick()
{ ; }

bool Tick::operator<( const Tick& other ) const
{
	const double EPSILON = 1e-9;

	if ( dt_ < other.dt_ ) return 1;
		if ( fabs( 1.0 - dt_ / other.dt_ ) < EPSILON && 
			index_ < other.index_ )
			return 1;
	return 0;
}


bool Tick::operator==( const Tick& other ) const
{
	return ( dt_ == other.dt_ && index_ == other.index_ );
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
 * set and get Path are problematic. Their goal is to assign the 
 * targets for this Tick. As framed, they fit in with the older
 * GENESIS syntax. For now, put in dummy functions.
void Tick::setPath( string v )
{
	path_ = v;
	// Something here to change the messages
}

string Tick::getPath() const
{
	return path_;
}
 */

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void Tick::destroy( Eref e, const Qinfo* q )
{
	;
}


///////////////////////////////////////////////////
// Virtual function definitions for actually sending out the 
// process and reinit calls.
///////////////////////////////////////////////////

void Tick::setIndex( unsigned int index ) 
{
	index_ = index;
}

void Tick::setElement( const Element* e ) 
{
	ticke_ = e;
}

/**
 * This sends out the call to reinit objects.
void Tick::reinit( const Eref& e, ProcInfo* info ) const
{
	info->dt = dt_;
	info->currTime = 0;

	assert( index_ < sizeof( reinitVec() ) / sizeof( SrcFinfo* ) );
	BindIndex b = reinitVec()[ index_ ]->getBindIndex();
	const vector< MsgFuncBinding >* m = e.element()->getMsgAndFunc( b );
	for ( vector< MsgFuncBinding >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		// Element->dataHandler keeps track of which entry needs to be
		// updated by which thread.
		Msg::getMsg( i->mid )->process( info, i->fid ); 
	}
}
 */

////////////////////////////////////////////////////////////////////////
// New version here
////////////////////////////////////////////////////////////////////////

/**
 * This function is called to advance this one tick through one 'process'
 * cycle. It is called in parallel on multiple threads, and the thread
 * info is in the ProcInfo. It is the job of the target Elements to
 * examine the ProcInfo and assign subsets of the object array to do
 * the process operation.
 */
void Tick::advance( ProcInfo* info ) const
{
	// March through Process calls for each scheduled Element.
	// Note that there is a unique BindIndex for each tick.
	// We preallocate 0 through 10 for this.
	assert( index_ < sizeof( processVec() ) / sizeof( SrcFinfo* ) );
	BindIndex b = processVec()[ index_ ]->getBindIndex();
	const vector< MsgFuncBinding >* m = ticke_->getMsgAndFunc( b );
	for ( vector< MsgFuncBinding >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		Msg::getMsg( i->mid )->process( info, i->fid ); 
	}
}

/**
 * This sends out the call to reinit objects.
 */
void Tick::reinit( ProcInfo* info ) const
{
	info->dt = dt_;
	info->currTime = 0;

	assert( index_ < sizeof( reinitVec() ) / sizeof( SrcFinfo* ) );
	BindIndex b = reinitVec()[ index_ ]->getBindIndex();
	const vector< MsgFuncBinding >* m = ticke_->getMsgAndFunc( b );
	for ( vector< MsgFuncBinding >::const_iterator i = m->begin();
		i != m->end(); ++i ) {
		// cout << info->nodeIndexInGroup << "." << info->threadIndexInGroup << ": reinit[" << index_ << "] binding = (" << i->mid << "," << i->fid << ")\n";
		// Element->dataHandler keeps track of which entry needs to be
		// updated by which thread.
		Msg::getMsg( i->mid )->process( info, i->fid ); 
	}
}
