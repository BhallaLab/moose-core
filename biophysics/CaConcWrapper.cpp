#include <math.h>
#include "header.h"
#include "CaConc.h"
#include "CaConcWrapper.h"


Finfo* CaConcWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"Ca", &CaConcWrapper::getCa, 
		&CaConcWrapper::setCa, "double" ),
	new ValueFinfo< double >(
		"CaBasal", &CaConcWrapper::getCaBasal, 
		&CaConcWrapper::setCaBasal, "double" ),
	new ValueFinfo< double >(	// for backward compat
		"Ca_base", &CaConcWrapper::getCaBasal, 
		&CaConcWrapper::setCaBasal, "double" ),
	new ValueFinfo< double >(
		"tau", &CaConcWrapper::getTau, 
		&CaConcWrapper::setTau, "double" ),
	new ValueFinfo< double >(
		"B", &CaConcWrapper::getB, 
		&CaConcWrapper::setB, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"concOut", &CaConcWrapper::getConcSrc, 
		"reinitIn, processIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"currentIn", &CaConcWrapper::currentFunc,
		&CaConcWrapper::getCurrentInConn, "" ),
	new Dest2Finfo< double, double >(
		"currentFractionIn", &CaConcWrapper::currentFractionFunc,
		&CaConcWrapper::getCurrentFractionInConn, "" ),
	new Dest1Finfo< double >(
		"increaseIn", &CaConcWrapper::increaseFunc,
		&CaConcWrapper::getIncreaseInConn, "" ),
	new Dest1Finfo< double >(
		"decreaseIn", &CaConcWrapper::decreaseFunc,
		&CaConcWrapper::getDecreaseInConn, "" ),
	new Dest1Finfo< double >(
		"basalIn", &CaConcWrapper::basalFunc,
		&CaConcWrapper::getBasalInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &CaConcWrapper::reinitFunc,
		&CaConcWrapper::getProcessConn, "concOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &CaConcWrapper::processFunc,
		&CaConcWrapper::getProcessConn, "concOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &CaConcWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo CaConcWrapper::cinfo_(
	"CaConc",
	"Upinder S. Bhalla, 2006, NCBS",
	"CaConc: Calcium concentration pool. Takes current from a channel\nand keeps track of calcium buildup and depletion by a\nsingle exponential process.",
	"Neutral",
	CaConcWrapper::fieldArray_,
	sizeof(CaConcWrapper::fieldArray_)/sizeof(Finfo *),
	&CaConcWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void CaConcWrapper::reinitFuncLocal(  )
{
			activation_ = 0.0;
			c_ = 0.0;
			Ca_ = CaBasal_;
			concSrc_.send( Ca_ );
}
void CaConcWrapper::processFuncLocal( ProcInfo info )
{
			double x = exp( -info->dt_ / tau_ );
			c_ = c_ * x + ( B_ * activation_ * tau_ )  * ( 1.0 - x );
			Ca_ = CaBasal_ + c_;
			concSrc_.send( Ca_ );
			activation_ = 0;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnCaConcLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CaConcWrapper, processConn_ );
	return reinterpret_cast< CaConcWrapper* >( ( unsigned long )c - OFFSET );
}

Element* currentFractionInConnCaConcLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( CaConcWrapper, currentFractionInConn_ );
	return reinterpret_cast< CaConcWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Creation function
///////////////////////////////////////////////////
Element* CaConcWrapper::create(
	const string& name, Element* pa, const Element* proto ) {
	const CaConcWrapper* p = dynamic_cast<const CaConcWrapper *>(proto);
	CaConcWrapper* ret = new CaConcWrapper(name);
	if ( p ) {
		ret->Ca_ = p->Ca_;
		ret->CaBasal_ = p->CaBasal_;
		ret->tau_ = p->tau_;
		ret->B_ = p->B_;
	}
	// if (p)... and so on. 
	return ret;
}
