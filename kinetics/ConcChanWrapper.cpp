/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include "header.h"
#include "ConcChan.h"
#include "ConcChanWrapper.h"

const double ConcChan::R = 8.3144;  
const double ConcChan::F = 96485.3415; 


Finfo* ConcChanWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"permeability", &ConcChanWrapper::getPermeability, 
		&ConcChanWrapper::setPermeability, "double" ),
	new ValueFinfo< double >(
		"n", &ConcChanWrapper::getN, 
		&ConcChanWrapper::setN, "double" ),
	new ValueFinfo< double >(
		"Vm", &ConcChanWrapper::getVm, 
		&ConcChanWrapper::setVm, "double" ),
	new ValueFinfo< double >(
		"ENernst", &ConcChanWrapper::getENernst, 
		&ConcChanWrapper::setENernst, "double" ),
	new ValueFinfo< int >(
		"valence", &ConcChanWrapper::getValence, 
		&ConcChanWrapper::setValence, "int" ),
	new ValueFinfo< double >(
		"temperature", &ConcChanWrapper::getTemperature, 
		&ConcChanWrapper::setTemperature, "double" ),
	new ValueFinfo< double >(
		"inVol", &ConcChanWrapper::getInVol, 
		&ConcChanWrapper::setInVol, "double" ),
	new ValueFinfo< double >(
		"outVol", &ConcChanWrapper::getOutVol, 
		&ConcChanWrapper::setOutVol, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc2Finfo< double, double >(
		"influxOut", &ConcChanWrapper::getInfluxSrc, 
		"processIn", 1 ),
	new SingleSrc2Finfo< double, double >(
		"effluxOut", &ConcChanWrapper::getEffluxSrc, 
		"processIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"influxIn", &ConcChanWrapper::influxFunc,
		&ConcChanWrapper::getInfluxConn, "", 1 ),
	new Dest1Finfo< double >(
		"effluxIn", &ConcChanWrapper::effluxFunc,
		&ConcChanWrapper::getEffluxConn, "", 1 ),
	new Dest1Finfo< double >(
		"nIn", &ConcChanWrapper::nFunc,
		&ConcChanWrapper::getNInConn, "" ),
	new Dest1Finfo< double >(
		"VmIn", &ConcChanWrapper::VmFunc,
		&ConcChanWrapper::getVmInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &ConcChanWrapper::reinitFunc,
		&ConcChanWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ConcChanWrapper::processFunc,
		&ConcChanWrapper::getProcessConn, "influxOut, effluxOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &ConcChanWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"influx", &ConcChanWrapper::getInfluxConn,
		"influxIn, influxOut" ),
	new SharedFinfo(
		"efflux", &ConcChanWrapper::getEffluxConn,
		"effluxIn, effluxOut" ),
};

const Cinfo ConcChanWrapper::cinfo_(
	"ConcChan",
	"Upinder S. Bhalla, 2006, NCBS",
	"ConcChan: Simple channel that permits molecules to cross a conc gradient.",
	"Neutral",
	ConcChanWrapper::fieldArray_,
	sizeof(ConcChanWrapper::fieldArray_)/sizeof(Finfo *),
	&ConcChanWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ConcChanWrapper::reinitFuncLocal(  )
{
			if ( influxConn_.nTargets() > 0 )
				Ftype1< double >::get( 
					influxConn_.target( 0 )->parent(), "volumeScale",
					outVolumeScale_ );
			if ( effluxConn_.nTargets() > 0 )
				Ftype1< double >::get( 
					effluxConn_.target( 0 )->parent(), "volumeScale",
					inVolumeScale_ );
			if ( outVolumeScale_ <= 0.0 )
				outVolumeScale_ = 1.0;
			if ( inVolumeScale_ <= 0.0 )
				inVolumeScale_ = 1.0;
			A_ = B_ = 0.0;
			nernstScale_ = R * temperature_ / ( F * valence_ );
}
// A is for influx, B is for efflux.
void ConcChanWrapper::processFuncLocal( ProcInfo info )
{
			if ( valence_ == 0 ) {
				A_ *= n_ * permeability_ / outVolumeScale_;
				B_ *= n_ * permeability_ / inVolumeScale_;
			influxSrc_.send( B_, A_ );
			effluxSrc_.send( A_, B_ );
				A_ = B_ = 0.0;
			} else {
				ENernst_ = nernstScale_ * log( B_ * outVolumeScale_ /
					( A_ * inVolumeScale_ ) );
				A_ = ( ENernst_ - Vm_ ) * n_ * permeability_ * valence_;
				B_ = 0.0;
			influxSrc_.send( A_, B_ );
			effluxSrc_.send( B_, A_ );
				A_ = 1.0; 
			}
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnConcChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ConcChanWrapper, processConn_ );
	return reinterpret_cast< ConcChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* influxConnConcChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ConcChanWrapper, influxConn_ );
	return reinterpret_cast< ConcChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* effluxConnConcChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ConcChanWrapper, effluxConn_ );
	return reinterpret_cast< ConcChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* nInConnConcChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ConcChanWrapper, nInConn_ );
	return reinterpret_cast< ConcChanWrapper* >( ( unsigned long )c - OFFSET );
}

Element* VmInConnConcChanLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ConcChanWrapper, VmInConn_ );
	return reinterpret_cast< ConcChanWrapper* >( ( unsigned long )c - OFFSET );
}

