#include "header.h"
#include "SpikeGen.h"
#include "SpikeGenWrapper.h"


Finfo* SpikeGenWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"threshold", &SpikeGenWrapper::getThreshold, 
		&SpikeGenWrapper::setThreshold, "double" ),
	new ValueFinfo< double >(
		"absoluteRefractoryPeriod", &SpikeGenWrapper::getAbsoluteRefractoryPeriod, 
		&SpikeGenWrapper::setAbsoluteRefractoryPeriod, "double" ),
	new ValueFinfo< double >(
		"amplitude", &SpikeGenWrapper::getAmplitude, 
		&SpikeGenWrapper::setAmplitude, "double" ),
	new ValueFinfo< double >(
		"state", &SpikeGenWrapper::getState, 
		&SpikeGenWrapper::setState, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"eventOut", &SpikeGenWrapper::getEventSrc, 
		"channelIn" ),
	new SingleSrc2Finfo< double, double >(
		"channelOut", &SpikeGenWrapper::getChannelSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, ProcInfo >(
		"channelIn", &SpikeGenWrapper::channelFunc,
		&SpikeGenWrapper::getChannelConn, "eventOut", 1 ),
	new Dest1Finfo< double >(
		"reinitIn", &SpikeGenWrapper::reinitFunc,
		&SpikeGenWrapper::getChannelConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"channel", &SpikeGenWrapper::getChannelConn,
		"channelOut, channelIn, reinitIn" ),
};

const Cinfo SpikeGenWrapper::cinfo_(
	"SpikeGen",
	"Upi Bhalla, NCBS, Feb 2006",
	"SpikeGen: Spike generator object with thresholding and refractory period",
	"Neutral",
	SpikeGenWrapper::fieldArray_,
	sizeof(SpikeGenWrapper::fieldArray_)/sizeof(Finfo *),
	&SpikeGenWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SpikeGenWrapper::channelFuncLocal( double V, ProcInfo info )
{
			double t = info->currTime_;
			if ( V > threshold_ && 
				t > lastEvent_ + absoluteRefractoryPeriod_ ) {
				eventSrc_.send( t );
				lastEvent_ = t;
				state_ = amplitude_;
			} else  {
				state_ = 0.0;
			}
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* channelConnSpikeGenLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( SpikeGenWrapper, channelConn_ );
	return reinterpret_cast< SpikeGenWrapper* >( ( unsigned long )c - OFFSET );
}

