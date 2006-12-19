#include "header.h"
#include "Job.h"
#include "JobWrapper.h"


Finfo* JobWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ReadOnlyValueFinfo< int >(
		"running", &JobWrapper::getRunning, "int" ),
	new ValueFinfo< int >(
		"doLoop", &JobWrapper::getDoLoop, &JobWrapper::setDoLoop, "int" ),
	new ValueFinfo< int >(
		"doTiming", &JobWrapper::getDoTiming, &JobWrapper::setDoTiming, "int" ),
	new ValueFinfo< double >(
		"realTimeInterval", &JobWrapper::getRealTimeInterval, &JobWrapper::setRealTimeInterval, "double" ),
	new ValueFinfo< int >(
		"priority", &JobWrapper::getPriority, &JobWrapper::setPriority, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &JobWrapper::getProcessSrc, "" ),
	new NSrc0Finfo(
		"triggerOut", &JobWrapper::getTriggerSrc, "" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"processIn", &JobWrapper::processFunc,
		&JobWrapper::getProcessInConn, "" ),
	new Dest0Finfo(
		"stopIn", &JobWrapper::stopFunc,
		&JobWrapper::getStopInConn, "" ),
	new Dest1Finfo< double >(
		"sleepIn", &JobWrapper::sleepFunc,
		&JobWrapper::getSleepInConn, "" ),
	new Dest0Finfo(
		"wakeIn", &JobWrapper::wakeFunc,
		&JobWrapper::getWakeInConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
};

const Cinfo JobWrapper::cinfo_(
	"Job",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"Job: Job class. Handles both repetitive and event-driven operations.",
	"Neutral",
	JobWrapper::fieldArray_,
	sizeof(JobWrapper::fieldArray_)/sizeof(Finfo *),
	&JobWrapper::create
);

///////////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////////

// Overridden by the ClockJob
void JobWrapper::processFuncLocal( ProcInfo info )
{
	while ( doLoop_ && !terminate_ )
	processSrc_.send( info );

	if ( info->currTime_ > wakeUpTime_ )
	processSrc_.send( info );
}

///////////////////////////////////////////////////////
// Synapse function definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* processInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( JobWrapper, processInConn_ );
	return reinterpret_cast< JobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stopJobInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( JobWrapper, stopInConn_ );
	return reinterpret_cast< JobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* sleepInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( JobWrapper, sleepInConn_ );
	return reinterpret_cast< JobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* wakeInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( JobWrapper, wakeInConn_ );
	return reinterpret_cast< JobWrapper* >( ( unsigned long )c - OFFSET );
}

