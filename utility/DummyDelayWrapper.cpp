/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DummyDelay.h"
#include "DummyDelayWrapper.h"


Finfo* DummyDelayWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"threshold", &DummyDelayWrapper::getThreshold, 
		&DummyDelayWrapper::setThreshold, "double" ),
	new ValueFinfo< int >(
		"delay", &DummyDelayWrapper::getDelay, 
		&DummyDelayWrapper::setDelay, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"spikeOut", &DummyDelayWrapper::getSpikeSrc, 
		"processIn" ),
	new NSrc2Finfo< double, double >(
		"spikeTimeOut", &DummyDelayWrapper::getSpikeTimeSrc, 
		"processIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"spikeIn", &DummyDelayWrapper::spikeFunc,
		&DummyDelayWrapper::getSpikeInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &DummyDelayWrapper::reinitFunc,
		&DummyDelayWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &DummyDelayWrapper::processFunc,
		&DummyDelayWrapper::getProcessConn, "spikeOut, spikeTimeOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &DummyDelayWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo DummyDelayWrapper::cinfo_(
	"DummyDelay",
	"",
	"DummyDelay: Stores a spike-amplitude and emits it after 'delay' steps",
	"Neutral",
	DummyDelayWrapper::fieldArray_,
	sizeof(DummyDelayWrapper::fieldArray_)/sizeof(Finfo *),
	&DummyDelayWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void DummyDelayWrapper::spikeFuncLocal( double amplitude )
{
			if( stepsRemaining_ != -1 || amplitude < threshold_ )
				return;
			amplitude_ = amplitude;
			stepsRemaining_ = delay_;
}
void DummyDelayWrapper::processFuncLocal( ProcInfo info )
{
			if( stepsRemaining_ == -1 )
				return;
			if( stepsRemaining_ == 0 ) {
			spikeSrc_.send( amplitude_ );
			spikeTimeSrc_.send( amplitude_, info->currTime_ );
			}
			--stepsRemaining_;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnDummyDelayLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( DummyDelayWrapper, processConn_ );
	return reinterpret_cast< DummyDelayWrapper* >( ( unsigned long )c - OFFSET );
}

