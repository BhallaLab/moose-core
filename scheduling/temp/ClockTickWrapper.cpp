#include "header.h"
typedef double ProcArg;
typedef int  SynInfo;
#include "ClockTick.h"
#include "ClockTickWrapper.h"


Finfo* ClockTickWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"dt", &ClockTickWrapper::getDt, &ClockTickWrapper::setDt, "double" ),
	new ValueFinfo< double >(
		"nextt", &ClockTickWrapper::getNextt, &ClockTickWrapper::setNextt, "double" ),
	new ValueFinfo< double >(
		"epsnextt", &ClockTickWrapper::getEpsnextt, &ClockTickWrapper::setEpsnextt, "double" ),
	new ValueFinfo< double >(
		"max_clocks", &ClockTickWrapper::getMax_clocks, &ClockTickWrapper::setMax_clocks, "double" ),
	new ValueFinfo< string >(
		"path", &ClockTickWrapper::getPath, &ClockTickWrapper::setPath, "string" ),
	new ValueFinfo< double >(
		"nclocks", &ClockTickWrapper::getNclocks, &ClockTickWrapper::setNclocks, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &ClockTickWrapper::getProcessSrc, 
		"processIn", 1 ),
	new NSrc0Finfo(
		"reinitOut", &ClockTickWrapper::getReinitSrc, 
		"reinitIn", 1 ),
	new NSrc1Finfo< double >(
		"passStepOut", &ClockTickWrapper::getPassStepSrc, 
		"checkStepIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"checkStepIn", &ClockTickWrapper::checkStepFunc,
		&ClockTickWrapper::getSolverStepConn, "passStepOut" ),
", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ClockTickWrapper::processFunc,
		&ClockTickWrapper::getClockConn, "processOut" ),
", 1 ),
	new Dest0Finfo(
		"reinitIn", &ClockTickWrapper::reinitFunc,
		&ClockTickWrapper::getClockConn, "reinitOut" ),
", 1 ),
	new Dest0Finfo(
		"reschedIn", &ClockTickWrapper::reschedFunc,
		&ClockTickWrapper::getClockConn, "" ),
", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ClockTickWrapper::getClockConn,
		"processIn, reinitIn, reschedIn" ),
	new SharedFinfo(
		"tick", &ClockTickWrapper::getTickConn,
		"processOut, reinitOut" ),
	new SharedFinfo(
		"solverStep", &ClockTickWrapper::getSolverStepConn,
		"passStepOut, checkStepIn" ),
};

const Cinfo ClockTickWrapper::cinfo_(
	"ClockTick",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"ClockTick: ClockTick class. Controls execution of objects on a given dt.",
	"Neutral",
	ClockTickWrapper::fieldArray_,
	sizeof(ClockTickWrapper::fieldArray_)/sizeof(Finfo *),
	&ClockTickWrapper::create
);

///////////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Synapse function definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* clockConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockTickWrapper, clockConn_ );
	return reinterpret_cast< ClockTickWrapper* >( ( unsigned long )c - OFFSET );
}

