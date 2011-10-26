/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include "moose.h"
#include "Interpol.h"
#include "Table.h"

/**
 * This is a reimplementation of the GENESIS table.
 * In MOOSE it is derived from the Interpol class.
 *
 */

const Cinfo* initTableCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
		RFCAST( &Table::process ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
		RFCAST( &Table::reinit ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
		sizeof( processShared ) / sizeof( Finfo* ),
		"This is a shared message to receive Process message from the scheduler.");

	static Finfo* inputRequestShared[] =
	{
			// Sends out the request. Issued from the process call.
		new SrcFinfo( "requestInput", Ftype0::global() ),
			// Handle the returned value.
                new DestFinfo( "handleInput", Ftype1< double >::global(),
				RFCAST( &Table::setInput ) ),
	};

	static Finfo* tableFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "input", ValueFtype1< double >::global(),
			GFCAST( &Table::getInput ),
			RFCAST( &Table::setInput )
		),
		new ValueFinfo( "output", ValueFtype1< double >::global(),
			GFCAST( &Table::getOutput ),
			RFCAST( &Table::setOutput )
		),
		new ValueFinfo( "step_mode", ValueFtype1< int >::global(),
			GFCAST( &Table::getStepMode ),
			RFCAST( &Table::setStepMode )
		),
		// Paste over silly old GENESIS inconsistency in naming.
		new ValueFinfo( "stepmode", ValueFtype1< int >::global(),
			GFCAST( &Table::getStepMode ),
			RFCAST( &Table::setStepMode )
		),
		new ValueFinfo( "stepsize", ValueFtype1< double >::global(),
			GFCAST( &Table::getStepsize ),
			RFCAST( &Table::setStepsize )
		),
		new ValueFinfo( "threshold", ValueFtype1< double >::global(),
			GFCAST( &Table::getStepsize ),
			RFCAST( &Table::setStepsize )
		),
		new ValueFinfo( "fname", ValueFtype1< string >::global(),
			GFCAST( &Table::getFname ),
			RFCAST( &Table::setFname )
		),
		new LookupFinfo( "tableLookup",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Table::getLookup ),
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "inputRequest", inputRequestShared, 
			sizeof( inputRequestShared ) / sizeof( Finfo* ),
			"This is a shared message to request and handle value  messages from fields." ),
		/*
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "inputRequest", inputRequestTypes, 2 ),
		*/
		
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "outputSrc", Ftype1< double >::global(),
			"Sends the output value every timestep." ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "msgInput", Ftype1< double >::global(), 
			RFCAST( &Table::setInput ),
			"Look up and interpolate value from table using x value. "
			"Put result in output, and also send out on outputSrc."
		),
		new DestFinfo( "sum", Ftype1< double >::global(), 
			RFCAST( &Table::sum ),
			"Sum this value onto the output field."
		),
		new DestFinfo( "prd", Ftype1< double >::global(), 
			RFCAST( &Table::prd ),
			"Multipy this value into the output field."
		),
		new DestFinfo( "input2", Ftype2< double, unsigned int >::global(), 
			RFCAST( &Table::input2 ),
			"Put value into table index specifiey by second arg."
		),

		new DestFinfo( "tabop", Ftype3< char, double, double >::global(), 
			RFCAST( &Table::tabop ),
			"Handle calls to perform operations on table entries.This is a backward compatibility feature. "
			"call table TABOP op [min max] was old syntax .Here we always require min and max. "
			"If both zero we assume that it is the same as the range of the table. "
			"The result of the TABOP is put into 'output' field of table."
		),

	};

	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] =
	{
		"Name", "Table",
		"Author","Upinder S. Bhalla, 2007, NCBS",
		"Description", "A table with a couple of message slots for adding dependencies "
				"on other fields. The table for the object is created using tabcreate. "
				"Does a table lookup with interpolation. Also permits one to modify "
				"the table with sum and product messages, so as to extend the "
				"dimensionality of the table.  The table element is a way of defining "
				"arbitrary input-output functions.  It is based on the interpol_struct "
				"described above, and provides the simplest form of access to it. "
				"Other values (possibly generated by other tables) may be summed or "
				"multiplied into the output value by means of messages, to permit "
				"pseudo-multidimensional functions to be generated by the table. "
				"Tables can also be used as function generators.",
	};
	static Cinfo tableCinfo(
	doc,
	sizeof( doc ) / sizeof( string ),	
	initInterpolCinfo(),
	tableFinfos,
	sizeof( tableFinfos ) / sizeof( Finfo * ),
	ValueFtype1< Table >::global(),
	schedInfo, 1
	);

	return &tableCinfo;
}

static const Cinfo* tableCinfo = initTableCinfo();

static const Slot outputSlot = 
	initTableCinfo()->getSlot( "outputSrc" );
static const Slot inputRequestSlot = 
	initTableCinfo()->getSlot( "inputRequest.requestInput" );

////////////////////////////////////////////////////////////////////
// Here we set up Table class functions
////////////////////////////////////////////////////////////////////
Table::Table()
	: input_( 0.0 ), output_( 0.0 ),
	stepSize_( 0.0 ), stepMode_( TAB_IO ),
	sy_( 0.0 ), py_( 1.0 ), lastSpike_( 0.0 ),
	counter_( 0 )
{ 
	;
}
////////////////////////////////////////////////////////////////////
// Here we set up Table value fields
////////////////////////////////////////////////////////////////////

void Table::setInput( const Conn* c, double input ) 
{
	static_cast< Table* >( c->data() )->input_ = input;
}
double Table::getInput( Eref e )
{
	return static_cast< Table* >( e.data() )->input_;
}

void Table::setOutput( const Conn* c, double output ) 
{
	static_cast< Table* >( c->data() )->output_ = output;
}
double Table::getOutput( Eref e )
{
	return static_cast< Table* >( e.data() )->output_;
}

void Table::setStepMode( const Conn* c, int value ) 
{
	static_cast< Table* >( c->data() )->stepMode_ = value;
}
int Table::getStepMode( Eref e )
{
	return static_cast< Table* >( e.data() )->stepMode_;
}

void Table::setStepsize( const Conn* c, double val ) 
{
	static_cast< Table* >( c->data() )->stepSize_ = val;
}
double Table::getStepsize( Eref e )
{
	return static_cast< Table* >( e.data() )->stepSize_;
}

void Table::setFname( const Conn* c, string val ) 
{
	static_cast< Table* >( c->data() )->fname_ = val;
}
string Table::getFname( Eref e )
{
	return static_cast< Table* >( e.data() )->fname_;
}

double Table::getLookup( Eref e, const double& x )
{
	return static_cast< Table* >( e.data() )->innerLookup( x );
}

////////////////////////////////////////////////////////////////////
// Here we set up Table Destination functions
////////////////////////////////////////////////////////////////////

void Table::sum( const Conn* c, double x )
{
	static_cast< Table* >( c->data() )->sy_ += x;
}

void Table::prd( const Conn* c, double x )
{
	static_cast< Table* >( c->data() )->py_ *= x;
}

void Table::input2( const Conn* c, double y, unsigned int x )
{
	Table* t = static_cast< Table* >( c->data() );
	t->setTableValue( y, x );
}

void Table::process( const Conn* c, ProcInfo p )
{
	static_cast< Table* >( c->data() )->
			innerProcess( c->target(), p );
}

void Table::reinit( const Conn* c, ProcInfo p )
{
	static_cast< Table* >( c->data() )->innerReinit( c, p );
}

void Table::tabop( const Conn* c, char op, double min, double max )
{
	static_cast< Table* >( c->data() )->innerTabop( op, min, max );
}

////////////////////////////////////////////////////////////////////
// Here we set up private Table class functions.
////////////////////////////////////////////////////////////////////

void Table::innerProcess( Eref e, ProcInfo p )
{
	double temp;
	unsigned long index;
	send0( e, inputRequestSlot );
	switch( stepMode_ ) {
		case TAB_IO :
			output_ = innerLookup( input_ ) * py_ + sy_ ;
			send1< double >( e, outputSlot, output_ );
			break;
		case TAB_LOOP:
			// Looks up values based on input and time. Loops around
			// when done.
			if ( fabs( stepSize_ ) < EPSILON ) {
				double looplen = xmax_ - xmin_;
				temp = input_ + p->currTime_;
				if ( fabs( looplen ) > EPSILON )
					temp -= looplen * ( static_cast< int >( temp / looplen ) );
			} else {
				temp = input_ + stepSize_;
				if ( temp > xmax_ )
					temp = xmin_;
				input_ = temp;
			}
			output_ = innerLookup( temp );
			send1< double >( e, outputSlot, output_ );
			// here we have a slight divergence from the old GENESIS
			// case, where if the table is not alloced it returns 
			// the SimulationTime. Bad idea.
			break;
		case TAB_ONCE:
			// Looks up values based on input and time. Does not
			// loop around.
			if ( fabs( stepSize_ ) < EPSILON ) {
				temp = input_ + p->currTime_;
			} else {
				temp = input_;
				input_ += stepSize_;
			}
			output_ = innerLookup( temp );
			send1< double >( e, outputSlot, output_ );
			break;
		case TAB_BUF:
			/**
			 * Fills a table with values one by one.
			 * Output value is current sample number, can be set to
			 * let us fill the table from any point.
			 * Unlike the GENESIS version, here we use vector
			 * operations so the size expands as needed. 
			 * Round off the output_ value which is a double.
			 */
			index = expandTable( e, output_ );
			table_[ index ] = input_;
			output_ += 1.0;
			xmax_ = output_;
			break;
		case TAB_BUF_TO_FILE:
			/**
			 * Buffers data till table is full, then appends it to
			 * a file and empties table so it can accumulate more.
			 * Replaces TAB_BUF in cases where there is so much data to
			 * store that it would fill up memory.
			 * Output value is current sample number, can be set to
			 * let us fill the table from any point.
			 */
			index = (static_cast< long >( output_ + 0.5 )) % table_.size();
			// cout << index << "," << table_.size() << "	";
			table_[ index ] = input_;
			output_ += 1.0;
			xmax_ = output_;
			if ( index == table_.size() - 1 ) { // dump it
				// cout << " Dumping " << fname_ << endl;
				innerPrint( fname_, true );
				for ( vector< double >::iterator i = table_.begin(); 
					i != table_.end(); ++i )
					*i = 0;
			}
			break;
		case TAB_DELAY:
			{
			/**
			 * Implements a delay line. Input is from a message,
			 * output is delayed by the size of the table.
			 * Here again semantics differ from GENESIS. In GENESIS
			 * the current sample number was in the 'input' field,
			 * which is silly and inconsistent with the usage of
			 * TAB_BUF. Here we can fix this. Also it makes things
			 * easier because the 'input' field is used for the
			 * incoming message.
			 */
			if ( table_.size() == 0 )
				break;
			if ( 1 + counter_ >= table_.size() ) {
				counter_ = 0;
			}
			output_ = table_[ counter_ ];
			table_[ counter_ ] = input_;
			++counter_;
			send1< double >( e, outputSlot, output_ );
			}
			break;
		case TAB_SPIKE:
			/**
			 * Fills the table with spike times, using a threshold
			 * Threshold value is the stepsize field. (bizarre, needed
			 * for backward compat. Here we also have an equivalent
			 * threshold field).
			 * Output value is current sample number, can be set
			 * Differs from GENESIS version because again we allow
			 * the table to expand as needed.
			 */
			if ( input_ > stepSize_ ) {
				if ( lastSpike_ <= stepSize_ ) { // Check for new spike
                       			index = expandTable( e, output_ );
					table_[ index ] = p->currTime_;
					output_ += 1.0;
					xmax_ = output_;
				}
			}
			lastSpike_ = input_;
			break;
		case TAB_FIELDS:
			/** 
			 * We don't implement this one, it is too bizarre.
			 */
			break;
		default:
			assert( 0 );
			break;
	};
	sy_ = 0.0;
	py_ = 1.0;
}

unsigned long Table::expandTable( Eref e, double size )
{
	// Do a rounding
	unsigned long index = static_cast< unsigned long >( 0.5 + size );
	if ( index > Interpol::MAX_DIVS ) {
		cout << "Error: " << e.e->name() << ": Table overflow\n";
		return 0;
	}
	if ( index > table_.capacity() ) {
		table_.reserve( index * 2 );
	}
	if ( table_.size() <= index )
		table_.resize( index + 1, 0.0 );
	return index;
}

void Table::innerReinit( const Conn* c, ProcInfo p )
{
	counter_ = 0;
	sy_ = 0.0;
	py_ = 1.0;

	vector< double >::iterator i;
	if ( stepMode_ == TAB_DELAY ) {
		for ( i = table_.begin(); i != table_.end(); i++ )
			*i = input_;
	}
	if ( stepMode_ == TAB_SPIKE ) {
		for ( i = table_.begin(); i != table_.end(); i++ )
			*i = 0.0;
	}
	if ( stepMode_ == TAB_BUF ) {
		xmax_ = output_ = 0.0;
		table_.resize( 0 );
	}
}

void Table::innerTabop( char op, double min, double max )
{
	unsigned int istart = 0;
	unsigned int istop = 0;
	if ( table_.size() == 0 ) {
		output_ = 0.0;
		input_ = -1;
		return;
	}

	if ( min == 0.0 && max == 0.0 ) {
		// Use the full table range here 
		istart = 0;
		istop = table_.size();
	} else {
		if ( min >= xmin_ && min < xmax_ )
			istart = static_cast< unsigned int >( ( min - xmin_ ) * invDx() );
		if ( max <= min ) {
			output_ = table_[ istart ];
			input_ = -1;
			return;
		}
			
		if ( max > xmax_ )
			istop = table_.size();
		else if ( max < xmin_ )
			istop = 0;
		else
			istop = static_cast< unsigned int >( ( max - xmin_ ) * invDx() );
	}

	doOp( op, istart, istop );
}

void Table::doOp( char op, unsigned int istart, unsigned int istop )
{
	double sum = 0.0;
	double temp = 0.0;
	unsigned int i;

	switch( op ) {
		case 'a': /* average */
			sum = 0.0;
			for( i = istart; i < istop; i++ )
				sum += table_[ i ];
			output_ = sum / ( istop-istart );
		break;
		case 'm': /* minimum */
			sum = table_[istart];
			for( i = istart + 1; i < istop; i++ ) {
				if ( sum > table_[i] )
					sum = table_[i];
			}
			output_ = sum;
		break;
		case 'M': /* maximum */
			sum = table_[ istart ];
			for( i = istart + 1; i < istop; i++ ) {
				if (sum < table_[i])
					sum = table_[i];
			}
			output_ = sum;
		break;
		case 'r': /* range */
			temp = sum = table_[ istart ];
			for( i = istart + 1; i < istop; i++ ) {
				if ( sum < table_[ i ] ) 
					sum = table_[ i ];
				if (temp > table_[ i ] )
					temp = table_[ i ];
			}
			output_ = temp - sum;
		break;
		case 's': /* slope - Just an easy hack for now */
			if ( istart == istop ) {
				output_ = 0.0;
			} else {
				sum = table_[ istart ];
				temp = table_[ istop ];
				output_ = ( temp - sum )/( istop - istart );
			}
		break;
		case 'S': /* Sumsq */
			sum = 0.0;
			for( i = istart + 1; i < istop; i++ )
				sum += table_[ i ] * table_[ i ];
			output_ = sqrt( sum );
		break;
		default: 
		break;
	}
}

#ifdef DO_UNIT_TESTS

#include "../element/Neutral.h"
static double calcWaveform( int i );

void testTable()
{
	cout << "\nTesting Table";
	unsigned int i;
	Element* t = Neutral::create( "Table", "t", Element::root()->id(),
		Id::scratchId() );
	Element* t2 = Neutral::create( "Table", "t2", Element::root()->id(),
		Id::scratchId() );
	SetConn c( t, 0 );
	SetConn c2( t2, 0 );
	ASSERT( t != 0, "created table" );

	ProcInfoBase pb;
	pb.dt_ = 1.0;
	
	ASSERT( 
		Eref( t ).add( "outputSrc", t2, "input" ), "making msg" );
		// t->findFinfo( "outputSrc" )-> add( t, t2, t2->findFinfo( "input" ) ), "making msg"
	set< double >( t, "xmin", 0.0 );
	set< double >( t, "xmax", 10.0 );
	set< int >( t, "xdivs", 10 );
	set< int >( t, "mode", 1 ); // Used to set interpolation on.
	for ( i = 0; i <= 10; i++ )
		lookupSet< double, unsigned int >( t, "table", 
				static_cast< double >( i * i ), i );
				
	// Testing simple table lookup
	bool ret;
	ret = set< int >( t, "step_mode", TAB_IO ); assert( ret );
	ret = set< double >( t, "input", 2.5 ); assert( ret );
	Table::process( &c, &pb );
	double v = 0.0;
	get< double >( t, "input", v );
	ASSERT( v == 2.5 , "TAB_IO" );
	get< double >( t, "output", v );
	ASSERT( fabs( v - 6.5 ) < 1e-8 , "TAB_IO" );
	get< double >( t2, "input", v );
	ASSERT( fabs( v - 6.5 ) < 1e-8 , "TAB_IO" );

	// Testing table lookup with addition of output
	ret = set< double >( t, "input", 3.5 ); assert( ret ); // (9 + 16) / 2 = 12.5
	ret = set< double >( t, "sum", 2.5 ); assert( ret ); // 12.5 + 2.5 = 15.0
	Table::process( &c, &pb );
	v = 0.0;
	get< double >( t, "output", v );
	ASSERT( fabs( v - 15.0 ) < 1e-8 , "sum" );
	get< double >( t2, "input", v );
	ASSERT( fabs( v - 15.0 ) < 1e-8 , "sum" );

	// Testing table lookup with scaling of output
	ret = set< double >( t, "input", 4.5 ); assert( ret ); // (16 + 25) / 2 = 20.5
	ret = set< double >( t, "prd", 2.0 ); assert( ret ); // 20.5 * 2.0 = 41.0
	Table::process( &c, &pb );
	v = 0.0;
	get< double >( t, "output", v );
	ASSERT( fabs( v - 41.0 ) < 1e-8 , "sum" );
	get< double >( t2, "input", v );
	ASSERT( fabs( v - 41.0 ) < 1e-8 , "sum" );
	
	// Testing two things: func generator, periodic (TAB_LOOP) and
	// buffering input like a shift-register ( TAB_BUF ) 
	ret = set< int >( t, "stepmode", TAB_LOOP ); assert( ret );
	// with stepsize = 0, we use simulation time to look up the value
	ret = set< double >( t, "input", 0.0 ); assert( ret );
	ret = set< double >( t, "stepsize", 0.0 ); assert( ret );
	ret = set< int >( t2, "stepmode", TAB_BUF ); assert( ret );
	ret = set< double >( t2, "output", 0.0 ); assert( ret );
	ret = set< double >( t2, "xmax", 1.0 ); assert( ret );
	ret = set< double >( t2, "xmin", 0.0 ); assert( ret );
	ret = set< int >( t2, "xdivs", 1 ); assert( ret );
	unsigned int k;
	for ( i = 0; i < 20; i++ ) {
		pb.currTime_ = pb.dt_ * i;
		Table::process( &c, &pb );
		get< double >( t, "output", v );
		k = i % 10;
		ASSERT( fabs( v - k * k ) < 1e-8 , "TAB_LOOP" );
		Table::process( &c2, &pb );
	}
	get< double >( t2, "output", v );
	ASSERT( fabs( v - 20.0 ) < 1e-8 , "TAB_BUF" );
	get< double >( t2, "xmax", v );
	ASSERT( fabs( v - 20.0 ) < 1e-8 , "TAB_BUF" );
	for ( i = 0; i < 20; i++ ) {
		ret = lookupGet< double, unsigned int >( t2, "table", v, i );
		assert( ret );
		k = i % 10;
		ASSERT( fabs( v - k * k ) < 1e-8 , "TAB_BUF" );
	}

	// Testing two more things: func generator, aperiodic (TAB_ONCE) and
	// delay line ( TAB_DELAY ) .
	// With the func generator we also change the mode to use a 
	// different stepsize of 0.5.
	// Now our original 10-sample waveform starts coming out at
	// t = 4, and is stretched into 20 samples.
	// This altered waveform is delayed by 10 samples
	// So the output should look like 0 for t < 14, 
	// ( (t - 12)/2 )^2 for later up to t < 34, but mind the 
	// interpolation.
	ret = set< int >( t, "stepmode", TAB_ONCE ); assert( ret );
	ret = set< double >( t, "input", -2.0 ); assert( ret );
	ret = set< double >( t, "stepsize", 0.5 ); assert( ret );
	ret = set< int >( t2, "stepmode", TAB_DELAY ); assert( ret );
	ret = set< double >( t2, "output", 0.0 ); assert( ret );
	ret = set< double >( t2, "input", 0.0 ); assert( ret );
	ret = set< double >( t2, "xmax", 1.0 ); assert( ret );
	ret = set< double >( t2, "xmin", 0.0 ); assert( ret );
	ret = set< int >( t2, "xdivs", 10 ); assert( ret );
	Table::reinit( &c2, &pb );

	double temp;
	for ( i = 0; i < 40; i++ ) {
		pb.currTime_ = pb.dt_ * i;
		Table::process( &c, &pb );
		get< double >( t, "input", v );
		ASSERT( fabs( v +1.5 - i * 0.5 ) < 1e-8 , "TAB_LOOP" );
		get< double >( t, "output", v );
		temp = calcWaveform( i );
		ASSERT( fabs( v - temp ) < 1e-8 , "TAB_LOOP" );
		Table::process( &c2, &pb );
		get< double >( t2, "output", v );
		temp = calcWaveform( static_cast< int >( i ) - 10 );
		ASSERT( fabs( v - temp ) < 1e-8 , "TAB_DELAY" );
	}

	// Testing spike detector.
	ret = set< int >( t, "stepmode", TAB_SPIKE ); assert( ret );
	ret = set< double >( t, "threshold", 8.5 ); assert( ret );
	ret = set< double >( t, "output", 0.0 ); assert( ret );
	Table::reinit( &c, &pb );
	for ( i = 0; i < 100; i++ ) {
		pb.currTime_ = pb.dt_ * i;

		// This is a spike on times 3, 8, 13...
		ret = set< double >( t, "input", 
				static_cast< double >( ( i % 5 ) * ( i % 5 ) ) );
		assert( ret );
		Table::process( &c, &pb );
	}

	for ( i = 0; i < 20; i++ ) {
		ret = lookupGet< double, unsigned int >( t, "table", temp, i );
		assert( ret );
		ASSERT( temp == static_cast< double >( i * 5 + 3 ),
					"TAB_SPIKE" );
	}

	// Testing inputRequest message and its ability to grab the
	// field of another object.
	Element* t3 = Neutral::create( "Table", "t3", Element::root()->id(),
		Id::scratchId() );
	ASSERT( 
		Eref( t3 ).add( "inputRequest", t, "output" ), 
			"making inputRequest msg"
		// t3->findFinfo( "inputRequest" )-> add( t3, t, t->findFinfo( "output" ) ), "making inputRequest msg"
	);
	ASSERT( set< double >( t, "output", 42.345 ) , "inputRequest" );
	SetConn c3( t3, 0 );
	Table::process( &c3, &pb );
	get< double >( t3, "input", v );
	ASSERT( v == 42.345, "inputRequest" );

	// Testing file print command
	ASSERT( set< string >( t, "print", "/tmp/test.txt" ), "print" );
	

	set( t, "destroy" );
	set( t2, "destroy" );
	set( t3, "destroy" );
}

double calcWaveform( int i )
{
	double temp;
	unsigned int k;
	if ( i < 4 ) {
		temp = 0.0;
	} else if ( i < 24 ) {
		k = ( i - 4 );
		if ( k % 2 == 0 ) {
			temp = k / 2;
			temp = temp * temp;
		} else {
			k = k / 2;
			temp = k * k + (k + 1) * ( k + 1 );
			temp = temp / 2.0;
		}
	} else {
		temp = 100;
	}
	return temp;
}

#endif // DO_UNIT_TESTS
