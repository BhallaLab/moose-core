/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#define mtrandf(l,h)     (l)==(h) ? (l) : mtrand() * ((h)-(l)) + (l)

#include "header.h"
#include <cmath>
#include "../randnum/randnum.h"
#include "RandomSpike.h"
#include "RandomSpikeWrapper.h"


Finfo* RandomSpikeWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"rate", &RandomSpikeWrapper::getRate, 
		&RandomSpikeWrapper::setRate, "double" ),
	new ValueFinfo< double >(
		"absoluteRefractoryPeriod", &RandomSpikeWrapper::getAbsoluteRefractoryPeriod, 
		&RandomSpikeWrapper::setAbsoluteRefractoryPeriod, "double" ),
	new ValueFinfo< double >(
		"state", &RandomSpikeWrapper::getState, 
		&RandomSpikeWrapper::setState, "double" ),
	new ValueFinfo< int >(
		"reset", &RandomSpikeWrapper::getReset, 
		&RandomSpikeWrapper::setReset, "int" ),
	new ValueFinfo< double >(
		"resetValue", &RandomSpikeWrapper::getResetValue, 
		&RandomSpikeWrapper::setResetValue, "double" ),
	new ValueFinfo< double >(
		"minAmp", &RandomSpikeWrapper::getMinAmp, 
		&RandomSpikeWrapper::setMinAmp, "double" ),
	new ValueFinfo< double >(
		"maxAmp", &RandomSpikeWrapper::getMaxAmp, 
		&RandomSpikeWrapper::setMaxAmp, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"stateOut", &RandomSpikeWrapper::getStateSrc, 
		"processIn" ),
	new NSrc2Finfo< double, double >(
		"stateTimeOut", &RandomSpikeWrapper::getStateTimeSrc, 
		"processIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &RandomSpikeWrapper::reinitFunc,
		&RandomSpikeWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< double >(
		"rateIn", &RandomSpikeWrapper::rateFunc,
		&RandomSpikeWrapper::getRateInConn, "" ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &RandomSpikeWrapper::processFunc,
		&RandomSpikeWrapper::getProcessConn, "stateOut, stateTimeOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &RandomSpikeWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo RandomSpikeWrapper::cinfo_(
	"RandomSpike",
	"",
	"RandomSpike: ",
	"Neutral",
	RandomSpikeWrapper::fieldArray_,
	sizeof(RandomSpikeWrapper::fieldArray_)/sizeof(Finfo *),
	&RandomSpikeWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void RandomSpikeWrapper::reinitFuncLocal(  )
{
			mtseed( 1 );  
			state_ = resetValue_;
			if ( rate_ <= 0.0 ) {
				lastEvent_ = 0.0;
				realRate_ = 0.0;
			} else {
				double p;
				while((p = mtrand()) == 0.0)    
					;
				lastEvent_ = log( p ) / rate_;
			}
}
void RandomSpikeWrapper::rateFuncLocal( double rate )
{
			rate_ = rate;
			double p = 1.0 - rate_ * absoluteRefractoryPeriod_;
			if ( p <= 0.0 )
				realRate_ = rate;
			else
		    		realRate_ = rate / p;
}
void RandomSpikeWrapper::processFuncLocal( ProcInfo info )
{
			double cTime = info->currTime_;
			if ( reset_ )
				state_ = resetValue_;
			if ( absoluteRefractoryPeriod_ > cTime - lastEvent_ )
				return;
			double p = realRate_ * 1.0;  
			if ( p >= mtrand() ) {
				lastEvent_ = cTime;
				state_ = mtrandf( minAmp_, maxAmp_ );
	  		}
			stateSrc_.send( state_ );
			stateTimeSrc_.send( state_, cTime );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnRandomSpikeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( RandomSpikeWrapper, processConn_ );
	return reinterpret_cast< RandomSpikeWrapper* >( ( unsigned long )c - OFFSET );
}

Element* rateInConnRandomSpikeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( RandomSpikeWrapper, rateInConn_ );
	return reinterpret_cast< RandomSpikeWrapper* >( ( unsigned long )c - OFFSET );
}

