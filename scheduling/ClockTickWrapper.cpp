#include "header.h"
#include "ClockTick.h"
#include "ClockTickWrapper.h"


Finfo* ClockTickWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"dt", &ClockTickWrapper::getDt, 
		&ClockTickWrapper::setDt, "double" ),
	new ValueFinfo< int >(
		"stage", &ClockTickWrapper::getStage, 
		&ClockTickWrapper::setStage, "int" ),
	new ValueFinfo< double >(
		"nextt", &ClockTickWrapper::getNextt, 
		&ClockTickWrapper::setNextt, "double" ),
	new ValueFinfo< double >(
		"epsnextt", &ClockTickWrapper::getEpsnextt, 
		&ClockTickWrapper::setEpsnextt, "double" ),
	new ValueFinfo< double >(
		"max_clocks", &ClockTickWrapper::getMax_clocks, 
		&ClockTickWrapper::setMax_clocks, "double" ),
	new ValueFinfo< string >(
		"path", &ClockTickWrapper::getPath, 
		&ClockTickWrapper::setPath, "string" ),
	new ValueFinfo< double >(
		"nclocks", &ClockTickWrapper::getNclocks, 
		&ClockTickWrapper::setNclocks, "double" ),
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
	new SingleSrc2Finfo< double, Conn* >(
		"dtOut", &ClockTickWrapper::getDtSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"checkStepIn", &ClockTickWrapper::checkStepFunc,
		&ClockTickWrapper::getSolverStepConn, "passStepOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ClockTickWrapper::processFunc,
		&ClockTickWrapper::getClockConn, "processOut", 1 ),
	new Dest0Finfo(
		"reinitIn", &ClockTickWrapper::reinitFunc,
		&ClockTickWrapper::getClockConn, "reinitOut", 1 ),
	new Dest0Finfo(
		"reschedIn", &ClockTickWrapper::reschedFunc,
		&ClockTickWrapper::getClockConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ClockTickWrapper::getClockConn,
		"processIn, reinitIn, reschedIn, dtOut" ),
	new SharedFinfo(
		"process", &ClockTickWrapper::getTickConn,
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
// Assumes the path is a wildcard field path, that is, has a
// finfo name as the last part of the path string.
void ClockTickWrapper::innerSetPath( const string& path )
{
	path_ = path;
	size_t pos = path.find_last_of("/");
	if ( pos == string::npos || pos == path.length()) {
		cerr << "Error:ClockTickWrapper::innerSetPath: no finfo name in" << path << "\n"; 
		return;
	}
	string finfoName = path.substr( pos + 1 );
	string pathHead = path.substr( 0, pos );

	vector< Element* > ret;
	vector< Element* >::iterator i;
	Element::startFind( pathHead, ret );
	processSrc_.dropAll();
	reinitSrc_.dropAll();

	Field src = field( "process" );
	for ( i = ret.begin(); i != ret.end(); i++ ) {
		Field dest = ( *i )->field( finfoName );
		src.add( dest );
	}
}

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

