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
static SrcFinfo1< double > *output()
{
	static SrcFinfo1< double > output( "output", 
			"Sends the output value every timestep."
	);
	return &output;
}

static SrcFinfo0 *requestInput()
{
	static SrcFinfo0 requestInput( "requestInput", 
			"Sends out the request. Issued from the process call."
	);
	return &requestInput;
}

static SrcFinfo1< vector< double >* >  *requestField()
{
	static SrcFinfo1< vector< double >* > requestField( "requestField", 
			"Sends out a request to a generic double field. "
			"Issued from the process call."
			"Works for any number of targets."
	);
	return &requestField;
}

/*
static DestFinfo* handleInput() {
	static DestFinfo handleInput( "handleInput", 
			"Handle the returned value, which is in a prepacked buffer.",
			new OpFunc1< Adaptor, PrepackedBuffer >( &Adaptor::handleBufInput )
	);
	return &handleInput;
}
*/

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
	static ReadOnlyValueFinfo< Adaptor, double > outputValue( 
			"outputValue",
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
	*/

	//////////////////////////////////////////////////////////////////////
	// Now set it all up.
	//////////////////////////////////////////////////////////////////////
	static Finfo* adaptorFinfos[] = 
	{
		&inputOffset,				// Value
		&outputOffset,				// Value
		&scale,						// Value
		&outputValue,				// ReadOnlyValue
		&input,						// DestFinfo
		output(),					// SrcFinfo
		requestInput(),				// SrcFinfo
		requestField(),				// SrcFinfo
		&proc,						// SharedFinfo
	//	&inputRequest,				// SharedFinfo
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
		counter_( 0 ),
		numRequestField_( 0 )
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

/*
void Adaptor::handleBufInput( PrepackedBuffer pb )
{
	assert( pb.dataSize() == 1 );
	double v = *reinterpret_cast< const double* >( pb.data() );
	sum_ += v;
	++counter_;
}
*/

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
	// static FuncId fid = handleInput()->getFid(); 
	requestInput()->send( e );
	if ( numRequestField_ > 0 ) {
			/*
		vector< double > vals( numRequestField_, 0.0 );
		vector< double* > args( numRequestField_ );
		for ( unsigned int i = 0; i < numRequestField_; ++i )
			args[i] = &vals[i];
		requestField()->sendVec( e, args );
		for ( unsigned int i = 0; i < numRequestField_; ++i ) {
			sum_ += vals[i];
		}
		counter_ += numRequestField_;
		*/
		vector< double > ret;
		requestField()->send( e, &ret );
		assert( ret.size() == numRequestField_ );
		for ( unsigned int i = 0; i < numRequestField_; ++i ) {
			sum_ += ret[i];
		}
		counter_ += numRequestField_;
	}
	innerProcess();
	output()->send( e, output_ );
}

void Adaptor::reinit( const Eref& e, ProcPtr p )
{
	numRequestField_ = e.element()->getMsgTargets( e.dataIndex(),
					requestField() ).size();
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
