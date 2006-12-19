/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include "Job.h"
#include "JobWrapper.h"
#include "ClockTickMsgSrc.h"
#include "ClockJob.h"
#include "ClockJobWrapper.h"


Finfo* ClockJobWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"runTime", &ClockJobWrapper::getRunTime, 
		&ClockJobWrapper::setRunTime, "double" ),
	new ReadOnlyValueFinfo< double >(
		"currentTime", &ClockJobWrapper::getCurrentTime, "double" ),
	new ValueFinfo< int >(
		"nSteps", &ClockJobWrapper::getNSteps, 
		&ClockJobWrapper::setNSteps, "int" ),
	new ReadOnlyValueFinfo< int >(
		"currentStep", &ClockJobWrapper::getCurrentStep, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &ClockJobWrapper::getProcessSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reschedOut", &ClockJobWrapper::getReschedSrc, 
		"reschedIn, resetIn", 1 ),
	new NSrc0Finfo(
		"reinitOut", &ClockJobWrapper::getReinitSrc, 
		"reinitIn, resetIn", 1 ),
	new NSrc1Finfo< Element* >(
		"schedNewObjectOut", &ClockJobWrapper::getSchedNewObjectSrc, 
		"schedNewObjectIn", 1 ),
	new NSrc0Finfo(
		"finishedOut", &ClockJobWrapper::getFinishedSrc, 
		"" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"startIn", &ClockJobWrapper::startFunc,
		&ClockJobWrapper::getStartInConn, "" ),
	new Dest2Finfo< ProcInfo, int >(
		"stepIn", &ClockJobWrapper::stepFunc,
		&ClockJobWrapper::getStepInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &ClockJobWrapper::reinitFunc,
		&ClockJobWrapper::getReinitInConn, "reinitOut" ),
	new Dest0Finfo(
		"reschedIn", &ClockJobWrapper::reschedFunc,
		&ClockJobWrapper::getReschedInConn, "reschedOut" ),
	new Dest0Finfo(
		"resetIn", &ClockJobWrapper::resetFunc,
		&ClockJobWrapper::getResetInConn, "reschedOut, reinitOut" ),
	new Dest2Finfo< double, Conn* >(
		"dtIn", &ClockJobWrapper::dtFunc,
		&ClockJobWrapper::getClockConn, "", 1 ),
	new Dest1Finfo< Element* >(
		"schedNewObjectIn", &ClockJobWrapper::schedNewObjectFunc,
		&ClockJobWrapper::getSchedNewObjectInConn, "schedNewObjectOut" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ClockJobWrapper::getClockConn,
		"processOut, reinitOut, reschedOut, schedNewObjectOut, dtIn" ),
};

const Cinfo ClockJobWrapper::cinfo_(
	"ClockJob",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"ClockJob: ClockJob class. Handles sequencing of operations in simulations",
	"Neutral",
	ClockJobWrapper::fieldArray_,
	sizeof(ClockJobWrapper::fieldArray_)/sizeof(Finfo *),
	&ClockJobWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ClockJobWrapper::startFuncLocal( ProcInfo info )
{
			cout << "starting run for " << runTime_ << " sec.\n";
			info->currTime_ = currentTime_;
			if ( tickSrc_ )
				tickSrc_->start( info, currentTime_ + runTime_,
					processSrc_ );
			currentTime_ = info->currTime_;
}
void ClockJobWrapper::stepFuncLocal( ProcInfo info, int nsteps )
{
			cout << "starting run for " << nsteps << " steps.\n";
			info->currTime_ = currentTime_;
			if ( tickSrc_ ) {
				runTime_ = nsteps * tickSrc_->dt();
				tickSrc_->start( info, currentTime_ + runTime_,
					processSrc_ );
			}
			currentTime_ = info->currTime_;
}
void ClockJobWrapper::reinitFuncLocal(  )
{
			currentTime_ = 0.0;
			currentStep_ = 0;
			reinitSrc_.send();
}
void ClockJobWrapper::reschedFuncLocal(  )
{
			if ( tickSrc_ )
				delete ( tickSrc_ );
			tickSrc_ = 0;
			vector< Field > f;
			Field kids = field( "processOut" );
			kids->dest( f, this );
			vector< ClockTickMsgSrc > ticks;
			for ( unsigned long i = 0; i < f.size(); i++ )
	 			ticks.push_back( 
					ClockTickMsgSrc( this, f[i].getElement() , i )
				);
			sort( ticks.begin(), ticks.end() );
			vector< ClockTickMsgSrc >::iterator j;
			ClockTickMsgSrc** currTick = &tickSrc_;
			ClockTickMsgSrc* prev = tickSrc_;
			for ( j = ticks.begin(); j != ticks.end(); j++ ) {
				*currTick = new ClockTickMsgSrc( *j );
				prev = *currTick;
				currTick = (*currTick)->next();
			}
			reschedSrc_.send();
}
void ClockJobWrapper::dtFuncLocal( double dt, Conn* tick )
{
			if ( tickSrc_ == 0 )
				return;
			unsigned long index = clockConn_.find( tick ); 
			tickSrc_->updateDt( dt, index ); 
			sortTicks(); 
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* startInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, startInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stepInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, stepInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reinitInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, reinitInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reschedInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, reschedInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* resetInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, resetInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* schedNewObjectInConnClockJobLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, schedNewObjectInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
void ClockJobWrapper::sortTicks( )
{
	ClockTickMsgSrc** i;
	ClockTickMsgSrc** j = &tickSrc_;
	bool swapping = 1;
	while ( swapping ) {
		swapping = 0;
		for ( i = (*j)->next(); *i != 0; i = (*i)->next() ) {
			if ( **i < **j ) {
				(*i)->swap( j );
				i = j;
				swapping = 1;
				break;
			} else {
				j = i;
			}
		}
	}
	tickSrc_->updateNextClockTime();
}
