#include "header.h"
#include "Interpol.h"
#include "InterpolWrapper.h"

const double Interpol::EPSILON = 1.0e-10;
const int Interpol::MAX_DIVS = 10000000; //Ten million points should do.

Finfo* InterpolWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"xmin", &InterpolWrapper::getXmin, 
		&InterpolWrapper::setXmin, "double" ),
	new ValueFinfo< double >(
		"xmax", &InterpolWrapper::getXmax, 
		&InterpolWrapper::setXmax, "double" ),
	new ValueFinfo< int >(
		"xdivs", &InterpolWrapper::getXdivs, 
		&InterpolWrapper::setXdivs, "int" ),
	new ValueFinfo< int >(
		"mode", &InterpolWrapper::getMode, 
		&InterpolWrapper::setMode, "int" ),
	new ValueFinfo< int >(
		"calc_mode", &InterpolWrapper::getMode, 
		&InterpolWrapper::setMode, "int" ),
	new ValueFinfo< double >(
		"dx", &InterpolWrapper::getDx, 
		&InterpolWrapper::setDx, "double" ),
	new ValueFinfo< double >(
		"sy", &InterpolWrapper::getSy, 
		&InterpolWrapper::setSy, "double" ),
	new ArrayFinfo< double >(
		"table", &InterpolWrapper::getTable, 
		&InterpolWrapper::setTable, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"lookupOut", &InterpolWrapper::getLookupSrc, 
		"lookupIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"lookupIn", &InterpolWrapper::lookupFunc,
		&InterpolWrapper::getLookupInConn, "lookupOut" ),
	new Dest2Finfo< int, int >(
		"tabFillIn", &InterpolWrapper::tabFillFunc,
		0, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
};

const Cinfo InterpolWrapper::cinfo_(
	"Interpol",
	"Upinder S. Bhalla, 2005, NCBS",
	"Interpol: Interpolation class. Handles lookup of a y value from an\nx value, where the x value is a double. Can either use\ninterpolation or roundoff to the nearest index.",
	"Neutral",
	InterpolWrapper::fieldArray_,
	sizeof(InterpolWrapper::fieldArray_)/sizeof(Finfo *),
	&InterpolWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void InterpolWrapper::setTable(
	Element* e , unsigned long index, double value )
{
	InterpolWrapper* f = static_cast< InterpolWrapper* >( e );
	if ( f->table_.size() > index )
		f->table_[ index ] = value;
}

double InterpolWrapper::getTable(
	const Element* e , unsigned long index )
{
	const InterpolWrapper* f = static_cast< const InterpolWrapper* >( e );
	if ( f->table_.size() > index )
		return f->table_[ index ];
	return f->table_[ 0 ];
}

///////////////////////////////////////////////////
// Functions for the Interpol ( not for its wrapper )
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
/*
Element* gateConnInterpolLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( InterpolWrapper, gateConn_ );
	return reinterpret_cast< InterpolWrapper* >( ( unsigned long )c - OFFSET );
}
*/


Element* lookupInConnInterpolLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( InterpolWrapper, lookupInConn_ );
	return reinterpret_cast< InterpolWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Utility function for handling offsets
///////////////////////////////////////////////////

unsigned long InterpolOffset()
{
	static InterpolWrapper iw("temp");
	static Interpol* ip = &iw;
	static const unsigned long offset = 
		( unsigned long )( ip ) - ( unsigned long )(&iw);
	return offset;
}

/////////////////////////////////////////////////////////////////////
// Here we set up string conversions
/////////////////////////////////////////////////////////////////////

template<> string val2str< Interpol >( Interpol val )
{
	char line[200];
	int i;
	string ret = "Interpol:\nallocated	calc_mode	xdivs	xmin	xmax	dx\n";
	sprintf( line, "%d		%d		%d	%g	%g	%g\n",
		1, val.localGetMode(), val.localGetXdivs(), 
		val.localGetXmin(), val.localGetXmax(), val.localGetDx() );
	ret += line;
	for ( i = 0; i < 100 && i < val.localGetXdivs(); i ++ ) {
		if ( (i % 7) == 0 )
			sprintf ( line, "[%d]	%.5g", i, val.getTableValue( i ) );
		else if ( (i % 7) == 6 )
			sprintf ( line, "	%.5g\n", val.getTableValue( i ) );
		else 
			sprintf ( line, "	%.5g", val.getTableValue( i ) );
		ret += line;
	}
	if ( i < val.localGetXdivs() ) {
		sprintf( line, " ... %d entries not displayed\n", val.localGetXdivs() - i );
		ret += line;
	}
	return ret;
}

// Does some nifty parsing to assign values in the table.
// For now only the first of these is implemented
// - Double: Sets the entire table to a single value
// - { v1, v2 ... }: Sets the table up using initializer type strings
// - path: Assigns to the specified table.
template<> Interpol str2val< Interpol >( const string& s )
{
	if ( isdigit( s[0] ) ) {
		double value = atof( s.c_str() );
		Interpol A( 0, 0.0, 1.0 );
		A.setTableValue( value, 0 );
		return A;
	}
	cerr << "Interpol::str2val: Not yet implemented\n";
	return Interpol();
}

/////////////////////////////////////////////////////////////////////
