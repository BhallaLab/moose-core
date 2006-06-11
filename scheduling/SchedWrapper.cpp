#include "header.h"
#include "Sched.h"
#include "SchedWrapper.h"

Finfo* SchedWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &SchedWrapper::getProcessSrc, "" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< string, string >(
		"startIn", &SchedWrapper::startFunc,
		&SchedWrapper::getStartInConn, "" ),
	new Dest1Finfo< string >(
		"stopIn", &SchedWrapper::stopFunc,
		&SchedWrapper::getStopInConn, "" ),
};

const Cinfo SchedWrapper::cinfo_(
	"Sched",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"Sched: Scheduler root class. Controls multiple jobs for I/O and \nprocessing.",
	"Neutral",
	SchedWrapper::fieldArray_,
	sizeof(SchedWrapper::fieldArray_)/sizeof(Finfo *),
	&SchedWrapper::create
);

///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* startInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SchedWrapper, startInConn_ );
	return reinterpret_cast< SchedWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stopSchedInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SchedWrapper, stopInConn_ );
	return reinterpret_cast< SchedWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////////
// Function definitions
///////////////////////////////////////////////////////

// All well and good, but I really should call the send from an
// event loop. This means that the ProcInfo should be on a synapse-like
// array at the msg src.
void SchedWrapper::startFuncLocal(
	const string& job, const string& shell )
{
	ProcInfoBase p( shell );
	Field j( job );
	if ( j.good() ) { // Assign value to it
		// Field s = cinfo()->field( "processOut" );
		Ftype1< ProcInfo >::set( j.getElement(), j.operator->(), &p );
		/*
		s.setElement( this );
		if ( s.add( j ) ) {
			processSrc_.send( &p );
			return;
		}
		*/
	}
}
