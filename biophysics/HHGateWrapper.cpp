#include <math.h>
#include "header.h"
#include "../builtins/Interpol.h"
#include "../builtins/InterpolWrapper.h"
#include "HHGate.h"
#include "HHGateWrapper.h"


static Conn* getDummyConn( Element* e ) {
	return Finfo::dummyConn();
}

Finfo* HHGateWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"power", &HHGateWrapper::getPower, 
		&HHGateWrapper::setPower, "double" ),
	new ValueFinfo< double >(
		"state", &HHGateWrapper::getState, 
		&HHGateWrapper::setState, "double" ),
	new ValueFinfo< int >(
		"instant", &HHGateWrapper::getInstant, 
		&HHGateWrapper::setInstant, "int" ),
	new ObjFinfo< Interpol >(
		"A", &HHGateWrapper::getA,
		&HHGateWrapper::setA, &HHGateWrapper::lookupA, "Interpol"),
	new ObjFinfo< Interpol >(
		"B", &HHGateWrapper::getB,
		&HHGateWrapper::setB, &HHGateWrapper::lookupB, "Interpol"),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new Return2Finfo< double, double >(
		"gateOut", &HHGateWrapper::getGateMultiReturnConn, 
		"gateIn, reinitIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest3Finfo< double, double, double >(
		"gateIn", &HHGateWrapper::gateFunc,
		&HHGateWrapper::getGateConn, "gateOut", 1 ),
	new Dest3Finfo< double, double, int >(
		"reinitIn", &HHGateWrapper::reinitFunc,
		&HHGateWrapper::getGateConn, "gateOut", 1 ),
	// This is needed because the redirection for ObjFinfo does not
	// work for msgdests. Yet.
	new Dest2Finfo< int, int >(
		"tabFillIn", &HHGateWrapper::tabFillFunc,
		&getDummyConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"gate", &HHGateWrapper::getGateConn,
		"gateIn, gateOut, reinitIn" ),
};

const Cinfo HHGateWrapper::cinfo_(
	"HHGate",
	"Upinder S. Bhalla, 2005, NCBS",
	"HHGate: Gate for Hodkgin-Huxley type channels, equivalent to the\nm and h terms on the Na squid channel and the n term on K.\nThis takes the voltage and state variable from the channel,\ncomputes the new value of the state variable and a scaling,\ndepending on gate power, for the conductance. These two\nterms are sent right back in a message to the channel.",
	"Neutral",
	HHGateWrapper::fieldArray_,
	sizeof(HHGateWrapper::fieldArray_)/sizeof(Finfo *),
	&HHGateWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHGateWrapper::gateFuncLocal( Conn* c, 
	double v, double state, double dt )
{
	ReturnConn* rc = static_cast< ReturnConn* >( c );
	if ( instant_ ) {
		state = A_.doLookup( v ) / B_.doLookup( v );
	} else {
		double y = B_.doLookup( v );
		double x = exp( -y * dt );
		state = state * x + ( A_.doLookup( v ) / y ) * ( 1 - x );
	// This ugly construction returns the info back to sender.
	}
	reinterpret_cast< void ( * )( Conn*, double, double ) >(
		rc->recvFunc() )
		( rc->rawTarget(), state, takePower_( state ) );
}

void HHGateWrapper::reinitFuncLocal( Conn* c, double Vm, double power,
	int instant )
{
	power_ = power;
	instant_ = ( instant != 0 );
	ReturnConn* rc = static_cast< ReturnConn* >( c );
	if ( power_ == 0.0 )
		takePower_ = power0;
	else if ( power_ == 1.0 )
		takePower_ = power1;
	else if ( power_ == 2.0 )
		takePower_ = power2;
	else if ( power_ == 3.0 )
		takePower_ = power3;
	else if ( power_ == 4.0 )
		takePower_ = power4;
	else
		takePower_ = power0;
	double x = A_.doLookup( Vm );
	double y = B_.doLookup( Vm );
	double z = x / y;
	//gateSrc_.send( z, takePower_( z ) );
	reinterpret_cast< void ( * )( Conn*, double, double ) >(
		rc->recvFunc() )
		( rc->rawTarget(), z, takePower_( z ) );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
