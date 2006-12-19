/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// This variant of ClockTick supports 5 stages involved in 
// managing parallel messaging.
//
// Stage 0: post irecv for this tick.
// Stage 1: Call all processes that have outgoing data on this tick.
// Stage 2: Post send
// Stage 3: Call all processes that only have local data.
// Stage 4: Poll for posted irecvs, as they arrive, send their contents.
//          The poll process relies on return info from each postmaster
//
// Stage 0, 2, 4 pass only tick stage info.
// Stage 1 and 3 pass regular ProcInfo

// Should really happen automatically when mpp sees it is derived.

#include "header.h"
#include "ClockTick.h"
#include "ClockTickWrapper.h"
#include "ParTick.h"
#include "ParTickWrapper.h"


Finfo* ParTickWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"handleAsync", &ParTickWrapper::getHandleAsync, 
		&ParTickWrapper::setHandleAsync, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"outgoingProcessOut", &ParTickWrapper::getOutgoingProcessSrc, 
		"processIn", 1 ),
	new NSrc0Finfo(
		"outgoingReinitOut", &ParTickWrapper::getOutgoingReinitSrc, 
		"reinitIn", 1 ),
	new NSrc1Finfo< int >(
		"ordinalOut", &ParTickWrapper::getOrdinalSrc, 
		"", 1 ),
	new NSrc1Finfo< int >(
		"asyncOut", &ParTickWrapper::getAsyncSrc, 
		"processIn, pollAsyncIn", 1 ),
	new NSrc1Finfo< int >(
		"postIrecvOut", &ParTickWrapper::getPostIrecvSrc, 
		"processIn, reinitIn", 1 ),
	new NSrc1Finfo< int >(
		"postSendOut", &ParTickWrapper::getPostSendSrc, 
		"processIn, reinitIn", 1 ),
	new NSrc1Finfo< int >(
		"pollRecvOut", &ParTickWrapper::getPollRecvSrc, 
		"processIn, reinitIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"pollRecvIn", &ParTickWrapper::pollRecvFunc,
		&ParTickWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ParTickWrapper::processFunc,
		&ParTickWrapper::getClockConn, "outgoingProcessOut, asyncOut, postIrecvOut, postSendOut, pollRecvOut", 1 ),
	new Dest0Finfo(
		"reinitIn", &ParTickWrapper::reinitFunc,
		&ParTickWrapper::getClockConn, "outgoingReinitOut, postIrecvOut, postSendOut, pollRecvOut", 1 ),
	new Dest0Finfo(
		"pollAsyncIn", &ParTickWrapper::pollAsyncFunc,
		&ParTickWrapper::getPollAsyncInConn, "asyncOut" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ParTickWrapper::getClockConn,
		"processIn, reinitIn, reschedIn, schedNewObjectIn, dtOut" ),
	new SharedFinfo(
		"outgoingProcess", &ParTickWrapper::getOutgoingProcessConn,
		"outgoingProcessOut, outgoingReinitOut" ),
	new SharedFinfo(
		"parProcess", &ParTickWrapper::getParProcessConn,
		"ordinalOut, asyncOut, postIrecvOut, postSendOut, pollRecvOut, pollRecvIn" ),
};

const Cinfo ParTickWrapper::cinfo_(
	"ParTick",
	"Upinder S. Bhalla, Nov 2006, NCBS",
	"ParTick: ParTick class. Controls execution of objects on a given dt,\nin the context of parallel messaging. Coordinates local and\noff-node object execution along with special calls to the\npostmaster.",
	"ClockTick",
	ParTickWrapper::fieldArray_,
	sizeof(ParTickWrapper::fieldArray_)/sizeof(Finfo *),
	&ParTickWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ParTickWrapper::processFuncLocal( ProcInfo info )
{
			// cout << "In ParTickWrapper::processFuncLocal\n";
			if ( handleAsync_ )
				asyncSrc_.send( ordinal() );
			postIrecvSrc_.send( ordinal() );
			outgoingProcessSrc_.send( info );
			postSendSrc_.send( ordinal() );
			ClockTickWrapper::processFuncLocal( info );
			numArrived_ = 0;
			while ( numArrived_ < numPostMaster_ )
			pollRecvSrc_.send( ordinal() );
}
void ParTickWrapper::reinitFuncLocal(  )
{
			separateOutgoingTargets();
			numPostMaster_ = parProcessConn_.nTargets();
			postIrecvSrc_.send( ordinal() );
			outgoingReinitSrc_.send( );
			postSendSrc_.send( ordinal() );
			ClockTickWrapper::reinitFuncLocal( );
			numArrived_ = 0;
			while ( numArrived_ < numPostMaster_ )
			pollRecvSrc_.send( ordinal() );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* clockConnParTickLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ParTickWrapper, clockConn_ );
	return reinterpret_cast< ParTickWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void ParTickWrapper::separateOutgoingTargets( )
{
}
Element* ParTickWrapper::create(
	const string& name, Element* pa, const Element* proto )
{
	if ( pa->cinfo()->isA( Cinfo::find( "ClockJob" ) ) ) {
		Field clock = pa->field( "clock" );
		ParTickWrapper* ret = new ParTickWrapper( name );
		ret->assignOrdinal();
		Field tick = ret->field( "clock" );
		clock.add( tick );
		Field parProcess = ret->field( "parProcess" );
		vector< Element* > postmasters;
		vector< Element* >::iterator i;
		Element::wildcardFind(
			"/postmasters/#[TYPE=PostMaster]", postmasters );
		//cout << "ParTickWrapper::create for " << name << ": Found " <<
		  //   postmasters.size() << " postmasters\n";
		for ( i = postmasters.begin(); i != postmasters.end(); i++ ) {
			Field f( *i, "parProcess" );
			// cout << "starting add\n";
			parProcess.add( f );
			// if ( parProcess.add( f ) )
				// cout << "did add for " << f.name() << endl ;
				
		}
		ret->ordinalSrc_.send( ret->ordinal() );
		return ret;
	};
	return 0;
}
