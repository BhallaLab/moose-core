#include "header.h"
typedef double ProcArg;
typedef int  SynInfo;
#include "MyClass.h"
#include "MyClassWrapper.h"

const double MyClass::pi_ = 3.1415926535 ;

Finfo* MyClassWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"Vm", &MyClassWrapper::getVm, &MyClassWrapper::setVm, "double" ),
	new ValueFinfo< double >(
		"Cm", &MyClassWrapper::getCm, &MyClassWrapper::setCm, "double" ),
	new ValueFinfo< double >(
		"Rm", &MyClassWrapper::getRm, &MyClassWrapper::setRm, "double" ),
	new ReadOnlyValueFinfo< double >(
		"pi", &MyClassWrapper::getPi, "double" ),
	new ReadOnlyValueFinfo< double >(
		"Ra", &MyClassWrapper::getRa, "double" ),
	new ValueFinfo< double >(
		"inject", &MyClassWrapper::getInject, &MyClassWrapper::setInject, "double" ),
	new ArrayFinfo< double >(
		"coords", &MyClassWrapper::getCoords, &MyClassWrapper::setCoords, "double" ),
	new ArrayFinfo< double >(
		"values", &MyClassWrapper::getValues, &MyClassWrapper::setValues, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"axialOut", &MyClassWrapper::getAxialSrc, "processIn" ),
	new SingleSrc2Finfo< double, double >(
		"raxialOut", &MyClassWrapper::getRaxialSrc, "processIn" ),
	new NSrc2Finfo< double, ProcArg >(
		"channelOut", &MyClassWrapper::getChannelSrc, "processIn" ),
	new NSrc2Finfo< double, double >(
		"diffusion1Out", &MyClassWrapper::getDiffusion1Src, "" ),
	new SingleSrc2Finfo< double, double >(
		"diffusion2Out", &MyClassWrapper::getDiffusion2Src, "" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"axialIn", &MyClassWrapper::axialFunc,
		&MyClassWrapper::getDistalConn, "" ),
	new Dest2Finfo< double, double >(
		"raxialIn", &MyClassWrapper::raxialFunc,
		&MyClassWrapper::getProximalConn, "" ),
	new Dest2Finfo< double, double >(
		"channelIn", &MyClassWrapper::channelFunc,
		&MyClassWrapper::getChannelInConn, "" ),
	new Dest2Finfo< double, double >(
		"diffusion2In", &MyClassWrapper::diffusion2Func,
		&MyClassWrapper::getDistalConn, "" ),
	new Dest2Finfo< double, double >(
		"diffusion1In", &MyClassWrapper::diffusion1Func,
		&MyClassWrapper::getProximalConn, "" ),
	new Dest1Finfo< ProcArg >(
		"processIn", &MyClassWrapper::processFunc,
		&MyClassWrapper::getProcessConn, "axialOut, raxialOut, channelOut" ),
	new Dest0Finfo(
		"resetIn", &MyClassWrapper::resetFunc,
		&MyClassWrapper::getProcessConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
	new ArrayFinfo< int >(
		"inhibValue", &MyClassWrapper::getInhibValue,
		&MyClassWrapper::setInhibValue, "int" ),
	new Synapse1Finfo< double >(
		"inhibIn", &MyClassWrapper::inhibFunc,
		&MyClassWrapper::getInhibConn, &MyClassWrapper::newInhibConn, "" ),

	/*
	new ArrayFinfo< SynInfo >(
		"exciteValue", &MyClassWrapper::getExciteValue,
		&MyClassWrapper::setExciteValue, "" ),
		*/
	new Synapse1Finfo< double >(
		"exciteIn", &MyClassWrapper::exciteFunc,
		&MyClassWrapper::getExciteConn, &MyClassWrapper::newExciteConn, "" ),

///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"proximal", &MyClassWrapper::getProximalConn,
		"axialOut, raxialIn, diffusion1Out, diffusion1In" ),
	new SharedFinfo(
		"distal", &MyClassWrapper::getDistalConn,
		"axialIn, raxialOut, diffusion2In, diffusion2Out" ),
	new SharedFinfo(
		"process", &MyClassWrapper::getProcessConn,
		"processIn, resetIn" ),
};

const Cinfo MyClassWrapper::cinfo_(
	"MyClass",
	"Uma the Programmer, today's date, Uma's Institution.",
	"MyClass: This is a test class. If it were a real class it would \nhave been much the same.",
	"Neutral",
	MyClassWrapper::fieldArray_,
	sizeof(MyClassWrapper::fieldArray_)/sizeof(Finfo *),
	&MyClassWrapper::create
);

///////////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////////

void MyClassWrapper::setCoords(
	Element* e , unsigned long index, double value )
{
	MyClassWrapper* f = static_cast< MyClassWrapper* >( e );
	if ( f->coords_.size() > index )
		f->coords_[ index ] = value;
}

double MyClassWrapper::getCoords(
	const Element* e , unsigned long index )
{
	const MyClassWrapper* f = static_cast< const MyClassWrapper* >( e );
	if ( f->coords_.size() > index )
		return f->coords_[ index ];
	return f->coords_[ 0 ];
}

void MyClassWrapper::setValues(
	Element* e , unsigned long index, double value )
{
	MyClassWrapper* f = static_cast< MyClassWrapper* >( e );
	if ( f->values_.size() > index )
		f->values_[ index ] = value;
}

double MyClassWrapper::getValues(
	const Element* e , unsigned long index )
{
	const MyClassWrapper* f = static_cast< const MyClassWrapper* >( e );
	if ( f->values_.size() > index )
		return f->values_[ index ];
	return f->values_[ 0 ];
}

///////////////////////////////////////////////////////
// Synapse function definitions
///////////////////////////////////////////////////////
void MyClassWrapper::setInhibValue(
	Element* e , unsigned long index, int value )
{
	MyClassWrapper* f = static_cast< MyClassWrapper* >( e );
	if ( f->inhibConn_.size() > index )
		f->inhibConn_[ index ]->value_ = value;
}

int MyClassWrapper::getInhibValue(
	const Element* e , unsigned long index )
{
	const MyClassWrapper* f = static_cast< const MyClassWrapper* >( e );
	if ( f->inhibConn_.size() > index )
		return f->inhibConn_[ index ]->value_;
	return f->inhibConn_[ 0 ]->value_;
}

void MyClassWrapper::inhibFunc( Conn* c, double delay )
{
	//SynConn< int >* s = static_cast< SynConn< int >* >( c );
	//MyClassWrapper* temp = static_cast< MyClassWrapper* >( c->parent() );
	// Here we do the synaptic function
	;
}

unsigned long MyClassWrapper::newInhibConn( Element* e ) {
	MyClassWrapper* temp = static_cast < MyClassWrapper* >( e );
	SynConn< int >* s = new SynConn< int >( e );
	temp->inhibConn_.push_back( s );
	return temp->inhibConn_.size( ) - 1;
 }
void MyClassWrapper::setExciteValue(
	Element* e , unsigned long index, SynInfo value )
{
	MyClassWrapper* f = static_cast< MyClassWrapper* >( e );
	if ( f->exciteConn_.size() > index )
		f->exciteConn_[ index ]->value_ = value;
}

SynInfo MyClassWrapper::getExciteValue(
	const Element* e , unsigned long index )
{
	const MyClassWrapper* f = static_cast< const MyClassWrapper* >( e );
	if ( f->exciteConn_.size() > index )
		return f->exciteConn_[ index ]->value_;
	return f->exciteConn_[ 0 ]->value_;
}

void MyClassWrapper::exciteFunc( Conn* c, double delay )
{
	// SynConn< SynInfo >* s = static_cast< SynConn< SynInfo >* >( c );
	// MyClassWrapper* temp = static_cast< MyClassWrapper* >( c->parent() );
	// Here we do the synaptic function
	;
}

unsigned long MyClassWrapper::newExciteConn( Element* e ) {
	MyClassWrapper* temp = static_cast < MyClassWrapper* >( e );
	SynConn< SynInfo >* s = new SynConn< SynInfo >( e );
	temp->exciteConn_.push_back( s );
	return temp->exciteConn_.size( ) - 1;
 }
///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* distalConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET( MyClassWrapper, distalConn_ );
	/*
		(unsigned long) ( &MyClassWrapper::distalConn_ );
	*/
	return reinterpret_cast< MyClassWrapper* >( ( unsigned long )c - OFFSET );
}

Element* processConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET( MyClassWrapper, processConn_ );
	//	(unsigned long) ( &MyClassWrapper::processConn_ );
	return reinterpret_cast< MyClassWrapper* >( ( unsigned long )c - OFFSET );
}

