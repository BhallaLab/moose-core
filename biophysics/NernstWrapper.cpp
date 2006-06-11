#include <math.h>
#include "header.h"
#include "Nernst.h"
#include "NernstWrapper.h"

const double Nernst::R_OVER_F = 8.6171458e-5;
const double Nernst::ZERO_CELSIUS = 273.15;

Finfo* NernstWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	/*
	new ValueFinfo< double >(
		"E", 
		&NernstWrapper::getE,
		&NernstWrapper::setE,
		"double" ),
		*/
	new ReadOnlyValueFinfo< double >(
		"E", &NernstWrapper::getE, "double" ),
	new ValueFinfo< double >(
		"Temperature", &NernstWrapper::getTemperature, 
		&NernstWrapper::setTemperature, "double" ),
	new ValueFinfo< int >(
		"valence", &NernstWrapper::getValence, 
		&NernstWrapper::setValence, "int" ),
	new ValueFinfo< double >(
		"Cin", &NernstWrapper::getCin, 
		&NernstWrapper::setCin, "double" ),
	new ValueFinfo< double >(
		"Cout", &NernstWrapper::getCout, 
		&NernstWrapper::setCout, "double" ),
	new ValueFinfo< double >(
		"scale", &NernstWrapper::getScale, 
		&NernstWrapper::setScale, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"EOut", &NernstWrapper::getESrc, 
		"CinIn, CoutIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"CinIn", &NernstWrapper::CinFunc,
		&NernstWrapper::getCinInConn, "EOut" ),
	new Dest1Finfo< double >(
		"CoutIn", &NernstWrapper::CoutFunc,
		&NernstWrapper::getCoutInConn, "EOut" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
};

const Cinfo NernstWrapper::cinfo_(
	"Nernst",
	"Upinder S. Bhalla, 2006, NCBS",
	"Nernst: Calculates Nernst potential for a given ion based on \nCin and Cout, the inside and outside concentrations.\nImmediately sends out the potential to all targets.",
	"Neutral",
	NernstWrapper::fieldArray_,
	sizeof(NernstWrapper::fieldArray_)/sizeof(Finfo *),
	&NernstWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void NernstWrapper::CinFuncLocal( double conc )
{
			Cin_ = conc;
			E_ = factor_ * log( Cout_ / Cin_ );
			ESrc_.send( E_ );
}
void NernstWrapper::CoutFuncLocal( double conc )
{
			Cout_ = conc;
			E_ = factor_ * log( Cout_ / Cin_ );
			ESrc_.send( E_ );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* CinInConnNernstLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( NernstWrapper, CinInConn_ );
	return reinterpret_cast< NernstWrapper* >( ( unsigned long )c - OFFSET );
}

Element* CoutInConnNernstLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( NernstWrapper, CoutInConn_ );
	return reinterpret_cast< NernstWrapper* >( ( unsigned long )c - OFFSET );
}

