/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Adaptor.h"

/**
 * This is the adaptor class. It is used in interfacing different kinds
 * of solver with each other, especially for electrical to chemical
 * signeur models.
 */
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > *outputSrc()
{
	static SrcFinfo1< double > outputSrc( "outputSrc", 
			"Sends the output value every timestep."
	);
	return &outputSrc;
}

static SrcFinfo0 *requestInput()
{
	static SrcFinfo0 requestInput( "requestInput", 
			"Sends out the request. Issued from the process call."
	);
	return &requestInput;
}

static SrcFinfo1< FuncId >  *requestField()
{
	static SrcFinfo1< FuncId > requestField( "requestField", 
			"Sends out a request to a generic double field. "
			"Issued from the process call."
	);
	return &requestField;
}

static DestFinfo* handleInput() {
	static DestFinfo handleInput( "handleInput", 
			"Handle the returned value, which is in a prepacked buffer.",
			new OpFunc1< Adaptor, PrepackedBuffer >( &Adaptor::handleBufInput )
	);
	return &handleInput;
}

const Cinfo* Adaptor::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	static ValueFinfo< Adaptor, double > inputOffset( 
			"inputOffset",
			"Offset to apply to input message, before scaling",
			&Adaptor::setInputOffset,
			&Adaptor::getInputOffset
		);
	static ValueFinfo< Adaptor, double > outputOffset( 
			"outputOffset",
			"Offset to apply at output, after scaling",
			&Adaptor::setOutputOffset,
			&Adaptor::getOutputOffset
		);
	static ValueFinfo< Adaptor, double > scale( 
			"scale",
			"Scaling factor to apply to input",
			&Adaptor::setScale,
			&Adaptor::getScale
		);
	static ReadOnlyValueFinfo< Adaptor, double > output( 
			"output",
			"This is the linearly transformed output.",
			&Adaptor::getOutput
		);

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo input( 
			"input",
			"Input message to the adaptor. If multiple inputs are "
			"received, the system averages the inputs.",
		   	new OpFunc1< Adaptor, double >( &Adaptor::input )
		);
	/*
		new DestFinfo( "setup", 
			Ftype4< string, double, double, double >::global(), 
			RFCAST( &Adaptor::setup ),
			"Sets up adaptor in placeholder mode."
			"This is done when the kinetic model is yet to be built, "
			"so the adaptor is given the src/target molecule name as "
			"a placeholder. Later the 'build' function will complete "
			"setting up the adaptor.\n"
			"Args: moleculeName, scale, inputOffset, outputOffset. "
			"Note that the direction of the adaptor operation is given "
			"by whether the channel/Ca is connected as input or output."
		),
		new DestFinfo( "build", Ftype0::global(), 
			RFCAST( &Adaptor::build ),
			"Completes connection to previously specified molecule "
			"on kinetic model."
		),
		*/

	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	static  DestFinfo process( "process", 
				"Handles 'process' call",
			new ProcOpFunc< Adaptor>( &Adaptor::process )
	);
	static  DestFinfo reinit( "reinit", 
				"Handles 'reinit' call",
			new ProcOpFunc< Adaptor>( &Adaptor::reinit )
	);

	static Finfo* processShared[] =
	{
			&process, &reinit
	};
	static SharedFinfo proc( "proc", 
		"This is a shared message to receive Process message "
		"from the scheduler. ",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	/*
	static DestFinfo handleInput( "handleInput", 
			"Handle the returned value.",
			new OpFunc1< Adaptor, double >( &Adaptor::input )
	);
	*/

	static Finfo* inputRequestShared[] =
	{
		requestInput(),
		handleInput()
	};
	static SharedFinfo inputRequest( "inputRequest",
		"This is a shared message to request and handle value "
	   "messages from fields.",
		inputRequestShared, 
		sizeof( inputRequestShared ) / sizeof( Finfo* )
	);

	//////////////////////////////////////////////////////////////////////
	// Now set it all up.
	//////////////////////////////////////////////////////////////////////
	static Finfo* adaptorFinfos[] = 
	{
		&inputOffset,				// Value
		&outputOffset,				// Value
		&scale,						// Value
		&output,					// ReadOnlyValue
		&input,						// DestFinfo
		outputSrc(),				// SrcFinfo
		requestField(),				// SrcFinfo
		&proc,						// SharedFinfo
		&inputRequest,				// SharedFinfo
	};
	
	static string doc[] =
	{
		"Name", "Adaptor",
		"Author", "Upinder S. Bhalla, 2008, NCBS",
		"Description", "Averages and rescales values to couple different kinds of simulation",
	};

	static Cinfo adaptorCinfo(
		"Adaptor",
		Neutral::initCinfo(),
		adaptorFinfos,
		sizeof( adaptorFinfos ) / sizeof( Finfo * ),
		new Dinfo< Adaptor >(),
		doc,
		sizeof( doc ) / sizeof( string )
	);

	return &adaptorCinfo;
}

static const Cinfo* adaptorCinfo = Adaptor::initCinfo();

////////////////////////////////////////////////////////////////////
// Here we set up Adaptor class functions
////////////////////////////////////////////////////////////////////
Adaptor::Adaptor()
	:	
		output_( 0.0 ), 
		inputOffset_( 0.0 ), 
		outputOffset_( 0.0 ),
		scale_( 1.0 ),
		molName_( "" ),
		sum_( 0.0 ), 
		counter_( 0 )
{ 
	;
}
////////////////////////////////////////////////////////////////////
// Here we set up Adaptor value fields
////////////////////////////////////////////////////////////////////

void Adaptor::setInputOffset( double value ) 
{
	inputOffset_ = value;
}
double Adaptor::getInputOffset() const
{
	return inputOffset_;
}

void Adaptor::setOutputOffset( double value ) 
{
	outputOffset_ = value;
}
double Adaptor::getOutputOffset() const
{
	return outputOffset_;
}

void Adaptor::setScale( double value ) 
{
	scale_ = value;
}
double Adaptor::getScale() const
{
	return scale_;
}

double Adaptor::getOutput() const
{
	return output_;
}


////////////////////////////////////////////////////////////////////
// Here we set up Adaptor Destination functions
////////////////////////////////////////////////////////////////////

void Adaptor::input( double v )
{
	sum_ += v;
	++counter_;
}

void Adaptor::handleBufInput( PrepackedBuffer pb )
{
	assert( pb.dataSize() == 1 );
	double v = *reinterpret_cast< const double* >( pb.data() );
	sum_ += v;
	++counter_;
}

// separated out to help with unit tests.
void Adaptor::innerProcess()
{
	if ( counter_ == 0 ) { 
		output_ = outputOffset_;
	} else {
		output_ = outputOffset_ + 
			scale_ * ( ( sum_ / counter_ ) - inputOffset_ );
	}
	sum_ = 0.0;
	counter_ = 0;
}

void Adaptor::process( const Eref& e, ProcPtr p )
{
	static FuncId fid = handleInput()->getFid(); 
	requestInput()->send( e, p->threadIndexInGroup );
	requestField()->send( e, p->threadIndexInGroup, fid );
	innerProcess();
	outputSrc()->send( e, p->threadIndexInGroup, output_ );
}

void Adaptor::reinit( const Eref& e, ProcPtr p )
{
	process( e, p );
}

/*
void Adaptor::setup( const Conn* c, 
		string molName, double scale, 
		double inputOffset, double outputOffset )
{
	cout << "doing Adaptor setup on " << c->target().name() <<
		" with " << molName << " " <<
		scale << ", io " << inputOffset << ", oo " << 
		outputOffset << endl;
}

void Adaptor::build( const Conn* c )
{
}
*/
