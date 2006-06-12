/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ForwardEuler.h"
#include "ForwardEulerWrapper.h"


Finfo* ForwardEulerWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ReadOnlyValueFinfo< int >(
		"isInitialized", &ForwardEulerWrapper::getIsInitialized, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc0Finfo(
		"reinitOut", &ForwardEulerWrapper::getReinitSrc, 
		"reinitIn", 1 ),
	new SingleSrc2Finfo< vector< double >* , double >(
		"integrateOut", &ForwardEulerWrapper::getIntegrateSrc, 
		"processIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< vector< double >*  >(
		"allocateIn", &ForwardEulerWrapper::allocateFunc,
		&ForwardEulerWrapper::getIntegrateConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &ForwardEulerWrapper::processFunc,
		&ForwardEulerWrapper::getProcessConn, "integrateOut", 1 ),
	new Dest0Finfo(
		"reinitIn", &ForwardEulerWrapper::reinitFunc,
		&ForwardEulerWrapper::getProcessConn, "reinitOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"integrate", &ForwardEulerWrapper::getIntegrateConn,
		"integrateOut, allocateIn, reinitOut" ),
	new SharedFinfo(
		"process", &ForwardEulerWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo ForwardEulerWrapper::cinfo_(
	"ForwardEuler",
	"Upinder S. Bhalla, June 2006, NCBS",
	"ForwardEuler: Illustration and base class for setting up numerical solvers.\nThis is currently set up to work only with the Stoich class,\nwhich represents biochemical networks.\nThe goal is to have a standard interface so different\nsolvers can work with different kinds of calculation.",
	"Neutral",
	ForwardEulerWrapper::fieldArray_,
	sizeof(ForwardEulerWrapper::fieldArray_)/sizeof(Finfo *),
	&ForwardEulerWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void ForwardEulerWrapper::allocateFuncLocal( vector< double >*  y )
{
			y_ = y;
			if ( !isInitialized_ || yprime_.size() != y->size() )
				yprime_.resize( y->size(), 0.0 );
			isInitialized_ = 1;
}
void ForwardEulerWrapper::processFuncLocal( ProcInfo info )
{
		vector< double >::iterator i;
		vector< double >::const_iterator j = yprime_.begin();
		integrateSrc_.send( &yprime_, info->dt_ );
		for ( i = y_->begin(); i != y_->end(); i++ )
			*i += *j++;
		long currT = static_cast< long >( info->currTime_ );
		if ( info->currTime_ - currT < info->dt_ )
			cout << info->currTime_ << "    " << y_->front() << "\n";
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* integrateConnForwardEulerLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ForwardEulerWrapper, integrateConn_ );
	return reinterpret_cast< ForwardEulerWrapper* >( ( unsigned long )c - OFFSET );
}

Element* processConnForwardEulerLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ForwardEulerWrapper, processConn_ );
	return reinterpret_cast< ForwardEulerWrapper* >( ( unsigned long )c - OFFSET );
}

