/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Adaptor.h"

/**
 * This is the adaptor class. It is used in interfacing different kinds
 * of simulation with each other, especially for multiscale models and
 * for connecting between robots and ordinary simulations.
 */

const Cinfo* initAdaptorCinfo()
{
	/** 
	 * This is a shared message to receive Process message from
	 * the scheduler. 
	 */
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
		RFCAST( &Adaptor::process ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
		RFCAST( &Adaptor::reinit ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ) );


	/** 
	 * This is a shared message to request and handle value
	 * messages from fields.
	 */
	static Finfo* inputRequestShared[] =
	{
			// Sends out the request. Issued from the process call.
		new SrcFinfo( "requestInput", Ftype0::global() ),
			// Handle the returned value.
	    new DestFinfo( "handleInput", Ftype1< double >::global(),
				RFCAST( &Adaptor::input ) ),
	};

	static Finfo* adaptorFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "inputOffset", ValueFtype1< double >::global(),
			GFCAST( &Adaptor::getInputOffset ),
			RFCAST( &Adaptor::setInputOffset )
		),
		new ValueFinfo( "outputOffset", ValueFtype1< double >::global(),
			GFCAST( &Adaptor::getOutputOffset ),
			RFCAST( &Adaptor::setOutputOffset )
		),
		new ValueFinfo( "scale", ValueFtype1< double >::global(),
			GFCAST( &Adaptor::getScale ),
			RFCAST( &Adaptor::setScale )
		),
		new ValueFinfo( "output", ValueFtype1< double >::global(),
			GFCAST( &Adaptor::getOutput ),
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "inputRequest", inputRequestShared, 
			sizeof( inputRequestShared ) / sizeof( Finfo* ) ),
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		/// Sends the output value every timestep.
		new SrcFinfo( "outputSrc", Ftype1< double >::global() ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		
		/**
		 * Averages inputs.
		 */
		new DestFinfo( "input", Ftype1< double >::global(), 
			RFCAST( &Adaptor::input )
		),
	};

	static SchedInfo schedInfo[] = { { process, 0, 0 } };

	static Cinfo adaptorCinfo(
	"Adaptor",
	"Upinder S. Bhalla, 2008, NCBS",
	"Averages and rescales values to couple different kinds of simulation",
	initNeutralCinfo(),
	adaptorFinfos,
	sizeof( adaptorFinfos ) / sizeof( Finfo * ),
	ValueFtype1< Adaptor >::global(),
		schedInfo, 1
	);

	return &adaptorCinfo;
}

static const Cinfo* adaptorCinfo = initAdaptorCinfo();

static const Slot outputSlot = 
	initAdaptorCinfo()->getSlot( "outputSrc" );
static const Slot inputRequestSlot = 
	initAdaptorCinfo()->getSlot( "inputRequest.requestInput" );

////////////////////////////////////////////////////////////////////
// Here we set up Adaptor class functions
////////////////////////////////////////////////////////////////////
Adaptor::Adaptor()
	:	
		output_( 0.0 ), 
		inputOffset_( 0.0 ), 
		outputOffset_( 0.0 ),
		scale_( 1.0 ),
		sum_( 0.0 ), 
		counter_( 0 )
{ 
	;
}
////////////////////////////////////////////////////////////////////
// Here we set up Adaptor value fields
////////////////////////////////////////////////////////////////////

void Adaptor::setInputOffset( const Conn* c, double value ) 
{
	static_cast< Adaptor* >( c->data() )->inputOffset_ = value;
}
double Adaptor::getInputOffset( Eref e )
{
	return static_cast< Adaptor* >( e.data() )->inputOffset_;
}

void Adaptor::setOutputOffset( const Conn* c, double value ) 
{
	static_cast< Adaptor* >( c->data() )->outputOffset_ = value;
}
double Adaptor::getOutputOffset( Eref e )
{
	return static_cast< Adaptor* >( e.data() )->outputOffset_;
}

void Adaptor::setScale( const Conn* c, double value ) 
{
	static_cast< Adaptor* >( c->data() )->scale_ = value;
}
double Adaptor::getScale( Eref e )
{
	return static_cast< Adaptor* >( e.data() )->scale_;
}

double Adaptor::getOutput( Eref e )
{
	return static_cast< Adaptor* >( e.data() )->output_;
}


////////////////////////////////////////////////////////////////////
// Here we set up Adaptor Destination functions
////////////////////////////////////////////////////////////////////

void Adaptor::input( const Conn* c, double v )
{
	Adaptor *a = static_cast< Adaptor* >( c->data() );
	a->sum_ += v;
	++a->counter_;
}


void Adaptor::process( const Conn* c, ProcInfo p )
{
	static_cast< Adaptor* >( c->data() )->
			innerProcess( c->target(), p );
}

void Adaptor::reinit( const Conn* c, ProcInfo p )
{
	static_cast< Adaptor* >( c->data() )->innerReinit( c, p );
}

////////////////////////////////////////////////////////////////////
// Here we set up private Adaptor class functions.
////////////////////////////////////////////////////////////////////

void Adaptor::innerProcess( Eref e, ProcInfo p )
{
	send0( e, inputRequestSlot );
	if ( counter_ == 0 ) { 
		output_ = outputOffset_;
	} else {
		output_ = outputOffset_ + ( ( sum_ / counter_ ) - inputOffset_ );
	}
	sum_ = 0.0;
	counter_ = 0;
	send1< double >( e, outputSlot, output_ );
}

void Adaptor::innerReinit( const Conn* c, ProcInfo p )
{
	sum_ = 0.0;
	counter_ = 0;
}
