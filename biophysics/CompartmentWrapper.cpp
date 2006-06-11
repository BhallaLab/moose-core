#include "header.h"
#include <math.h>
#include "../randnum/randnum.h"
#include "Compartment.h"
#include "CompartmentWrapper.h"

const double Compartment::EPSILON = 1.0e-15;

Finfo* CompartmentWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"Vm", &CompartmentWrapper::getVm, 
		&CompartmentWrapper::setVm, "double" ),
	new ValueFinfo< double >(
		"Em", &CompartmentWrapper::getEm, 
		&CompartmentWrapper::setEm, "double" ),
	new ValueFinfo< double >(
		"Cm", &CompartmentWrapper::getCm, 
		&CompartmentWrapper::setCm, "double" ),
	new ValueFinfo< double >(
		"Rm", &CompartmentWrapper::getRm, 
		&CompartmentWrapper::setRm, "double" ),
	new ValueFinfo< double >(
		"Ra", &CompartmentWrapper::getRa, 
		&CompartmentWrapper::setRa, "double" ),
	new ValueFinfo< double >(
		"Im", &CompartmentWrapper::getIm, 
		&CompartmentWrapper::setIm, "double" ),
	new ValueFinfo< double >(
		"Inject", &CompartmentWrapper::getInject, 
		&CompartmentWrapper::setInject, "double" ),
	new ValueFinfo< double >(
		"initVm", &CompartmentWrapper::getInitVm, 
		&CompartmentWrapper::setInitVm, "double" ),
	new ValueFinfo< double >(
		"diameter", &CompartmentWrapper::getDiameter, 
		&CompartmentWrapper::setDiameter, "double" ),
	new ValueFinfo< double >(
		"length", &CompartmentWrapper::getLength, 
		&CompartmentWrapper::setLength, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc2Finfo< double, ProcInfo >(
		"channelOut", &CompartmentWrapper::getChannelSrc, 
		"reinitIn, processIn", 1 ),
	new NSrc1Finfo< double >(
		"axialOut", &CompartmentWrapper::getAxialSrc, 
		"initIn", 1 ),
	new NSrc2Finfo< double, double >(
		"raxialOut", &CompartmentWrapper::getRaxialSrc, 
		"initIn", 1 ),
	new NSrc1Finfo< double >(
		"channelReinitOut", &CompartmentWrapper::getChannelReinitSrc, 
		"reinitIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, double >(
		"channelIn", &CompartmentWrapper::channelFunc,
		&CompartmentWrapper::getChannelConn, "", 1 ),
	new Dest2Finfo< double, double >(
		"raxialIn", &CompartmentWrapper::raxialFunc,
		&CompartmentWrapper::getAxialConn, "", 1 ),
	new Dest1Finfo< double >(
		"axialIn", &CompartmentWrapper::axialFunc,
		&CompartmentWrapper::getRaxialConn, "", 1 ),
	new Dest1Finfo< double >(
		"injectIn", &CompartmentWrapper::injectFunc,
		&CompartmentWrapper::getInjectInConn, "" ),
	new Dest2Finfo< double, double >(
		"randinjectIn", &CompartmentWrapper::randinjectFunc,
		&CompartmentWrapper::getRandinjectInConn, "" ),
	new Dest1Finfo< ProcInfo >(
		"initIn", &CompartmentWrapper::initFunc,
		&CompartmentWrapper::getInitConn, "axialOut, raxialOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"dummyReinitIn", &CompartmentWrapper::dummyReinitFunc,
		&CompartmentWrapper::getInitConn, "", 1 ),
	new Dest0Finfo(
		"reinitIn", &CompartmentWrapper::reinitFunc,
		&CompartmentWrapper::getProcessConn, "channelOut, channelReinitOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &CompartmentWrapper::processFunc,
		&CompartmentWrapper::getProcessConn, "channelOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &CompartmentWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"init", &CompartmentWrapper::getInitConn,
		"initIn, dummyReinitIn" ),
	new SharedFinfo(
		"channel", &CompartmentWrapper::getChannelConn,
		"channelIn, channelOut, channelReinitOut" ),
	new SharedFinfo(
		"axial", &CompartmentWrapper::getAxialConn,
		"axialOut, raxialIn" ),
	new SharedFinfo(
		"raxial", &CompartmentWrapper::getRaxialConn,
		"axialIn, raxialOut" ),
};

const Cinfo CompartmentWrapper::cinfo_(
	"Compartment",
	"Upinder S. Bhalla, 2005, NCBS",
	"Compartment: Passive compartment from cable theory. Has hooks for adding\nion channels.",
	"Neutral",
	CompartmentWrapper::fieldArray_,
	sizeof(CompartmentWrapper::fieldArray_)/sizeof(Finfo *),
	&CompartmentWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void CompartmentWrapper::raxialFuncLocal( double Ra, double Vm )
{
			A_ += Vm / Ra;
			B_ += 1.0 / Ra;
			Im_ += ( Vm - Vm_ ) / Ra;
}
void CompartmentWrapper::axialFuncLocal( double Vm )
{
			A_ += Vm / Ra_;
			B_ += 1.0 / Ra_;
			Im_ += ( Vm - Vm_ ) / Ra_;
}
void CompartmentWrapper::randinjectFuncLocal( double prob, double I )
{
			if ( mtrand() < prob * dt_ ) {
				Im_ += I;
				A_ += I;
			}
}
void CompartmentWrapper::reinitFuncLocal( )
{
			if ( Rm_ > 0 )
				invRm_ = 1.0 / Rm_;
			else
				invRm_ = 1.0;
			Vm_ = initVm_;
			A_ = 0.0;
			B_ = invRm_;
			Im_ = 0.0;
			// dt_ = info->dt_;
			sumInject_ = 0.0;
			channelReinitSrc_.send( Vm_ );
			// channelSrc_.send( Vm_, info );
}
void CompartmentWrapper::processFuncLocal( ProcInfo info )
{
	dt_ = info->dt_;
	A_ += Inject_ + sumInject_ + Em_ * invRm_;
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * info->dt_ / Cm_ );
		Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
	} else {
		Vm_ += ( A_ - Vm_ * B_ ) * info->dt_ / Cm_;
	}
	A_ = 0.0;
	B_ = invRm_;
	Im_ = 0.0;
	sumInject_ = 0.0;
	channelSrc_.send( Vm_, info );
	axialSrc_.send( Vm_ );
	raxialSrc_.send( Ra_, Vm_ );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnCompartmentLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CompartmentWrapper, processConn_ );
	return reinterpret_cast< CompartmentWrapper* >( ( unsigned long )c - OFFSET );
}

Element* initConnCompartmentLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CompartmentWrapper, initConn_ );
	return reinterpret_cast< CompartmentWrapper* >( ( unsigned long )c - OFFSET );
}

Element* randinjectInConnCompartmentLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CompartmentWrapper, randinjectInConn_ );
	return reinterpret_cast< CompartmentWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Compartment creation
///////////////////////////////////////////////////
Element* CompartmentWrapper::create(
	const string& name, Element* pa, const Element* proto )
{
	CompartmentWrapper* ret = new CompartmentWrapper(name);
	const CompartmentWrapper* p = 
		dynamic_cast< const CompartmentWrapper* >( proto );
	if ( p ) {
		*( static_cast< Compartment* >( ret ) ) = *p;
	}
	return ret;
}
