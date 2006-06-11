#include <fstream>
#include "header.h"
#include "Plot.h"
#include "PlotWrapper.h"


Finfo* PlotWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"currTime", &PlotWrapper::getCurrTime, 
		&PlotWrapper::setCurrTime, "double" ),
	new ValueFinfo< string >(
		"plotName", &PlotWrapper::getPlotName, 
		&PlotWrapper::setPlotName, "string" ),
	new ValueFinfo< int >(
		"npts", &PlotWrapper::getNpts, 
		&PlotWrapper::setNpts, "int" ),
	new ValueFinfo< int >(
		"jagged", &PlotWrapper::getJagged, 
		&PlotWrapper::setJagged, "int" ),
	new ArrayFinfo< double >(
		"x", &PlotWrapper::getX, 
		&PlotWrapper::setX, "double" ),
	new ArrayFinfo< double >(
		"y", &PlotWrapper::getY, 
		&PlotWrapper::setY, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new SingleSrc0Finfo(
		"trigPlotOut", &PlotWrapper::getTrigPlotSrc, 
		"processIn", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest0Finfo(
		"reinitIn", &PlotWrapper::reinitFunc,
		&PlotWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &PlotWrapper::processFunc,
		&PlotWrapper::getProcessConn, "trigPlotOut", 1 ),
	new Dest1Finfo< double >(
		"trigPlotIn", &PlotWrapper::trigPlotFunc,
		&PlotWrapper::getTrigPlotInConn, "" ),
	new Dest1Finfo< double >(
		"plotIn", &PlotWrapper::plotFunc,
		&PlotWrapper::getPlotInConn, "" ),
	new Dest1Finfo< string >(
		"printIn", &PlotWrapper::printFunc,
		&PlotWrapper::getPrintInConn, "" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &PlotWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"trigPlot", &PlotWrapper::getTrigPlotConn,
		"trigPlotIn, trigPlotOut" ),
};

const Cinfo PlotWrapper::cinfo_(
	"Plot",
	"Upinder S. Bhalla, 2005, NCBS",
	"Plot: Stores and prints xy data, typically time-series.\nNormal mode of operation is triggered by an incoming process\nmessage. The plot then triggers a request for data (trigPlot).\nIt expects the y value as an immediate response. This is\nthen pushed onto the plot value vector.\nBackward compatibility mode accepts periodic data values at\nwhatever rate the sourcing object happens to be clocked at.\nThe Plot object ignores them unless it has just been primed\nby a process message. If it has been primed it pushes the value\nonto the vector.\nThe jagged flag changes how it handles incoming data. This\nflag is meant to be set when plotting stochastic data.\nIf the flag is set, the Plot object first checks if the\nvalue has changed. If so, it appends a data point using the\nold y value but the new time. Then it appends another data\npoint using the new y and time values. This produces plots\nwith sharp transitions like the process being modeled.\nAlso saves space given the output of a typical stochastic model.",
	"Neutral",
	PlotWrapper::fieldArray_,
	sizeof(PlotWrapper::fieldArray_)/sizeof(Finfo *),
	&PlotWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void PlotWrapper::setX(
	Element* e , unsigned long index, double value )
{
	PlotWrapper* f = static_cast< PlotWrapper* >( e );
	if ( f->x_.size() > index )
		f->x_[ index ] = value;
}

double PlotWrapper::getX(
	const Element* e , unsigned long index )
{
	const PlotWrapper* f = static_cast< const PlotWrapper* >( e );
	if ( f->x_.size() > index )
		return f->x_[ index ];
	return 0.0;
}

void PlotWrapper::setY(
	Element* e , unsigned long index, double value )
{
	PlotWrapper* f = static_cast< PlotWrapper* >( e );
	if ( f->y_.size() > index )
		f->y_[ index ] = value;
}

double PlotWrapper::getY(
	const Element* e , unsigned long index )
{
	const PlotWrapper* f = static_cast< const PlotWrapper* >( e );
	if ( f->y_.size() > index )
		return f->y_[ index ];
	return 0.0;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void PlotWrapper::reinitFuncLocal(  )
{
			x_.resize( 0 );
			y_.resize( 0 );
			currTime_ = 0.0;
			prime_ = 0;
}
void PlotWrapper::processFuncLocal( ProcInfo info )
{
			prime_ = 1;
			currTime_ = info->currTime_;
			trigPlotSrc_.send();
}
void PlotWrapper::trigPlotFuncLocal( double yval )
{
			if ( jagged_ ) {
				double lasty = ( y_.size() > 0 ) ? y_.back(): 0.0;
				if ( yval != lasty ) {
					x_.push_back( currTime_ );
					y_.push_back( lasty );
					x_.push_back( currTime_ );
					y_.push_back( yval );
				}
			} else {
				x_.push_back( currTime_ );
				y_.push_back( yval );
			}
}
void PlotWrapper::plotFuncLocal( double yval )
{
			if ( prime_ )
				trigPlotFuncLocal( yval );
			prime_ = 0;
}
void PlotWrapper::printFuncLocal( string fileName )
{
			ofstream fout( fileName.c_str(), ios::app );
			fout << "/newplot\n/plotname " << name() << "\n";
			unsigned long i;
			const unsigned int max = y_.size();
			for ( i = 0; i < max; i++ )
				fout << x_[ i ] << "	" << y_[ i ] << "\n";
			fout << "\n";
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnPlotLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PlotWrapper, processConn_ );
	return reinterpret_cast< PlotWrapper* >( ( unsigned long )c - OFFSET );
}

Element* trigPlotConnPlotLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PlotWrapper, trigPlotConn_ );
	return reinterpret_cast< PlotWrapper* >( ( unsigned long )c - OFFSET );
}

Element* trigPlotInConnPlotLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PlotWrapper, trigPlotInConn_ );
	return reinterpret_cast< PlotWrapper* >( ( unsigned long )c - OFFSET );
}

Element* plotInConnPlotLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PlotWrapper, plotInConn_ );
	return reinterpret_cast< PlotWrapper* >( ( unsigned long )c - OFFSET );
}

Element* printInConnPlotLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PlotWrapper, printInConn_ );
	return reinterpret_cast< PlotWrapper* >( ( unsigned long )c - OFFSET );
}

