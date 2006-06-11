#include <math.h>
#include "header.h"
#include "Molecule.h"
#include "MoleculeWrapper.h"

const double Molecule::EPSILON = 1.0e-15;

Finfo* MoleculeWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"nInit", &MoleculeWrapper::getNInit, 
		&MoleculeWrapper::setNInit, "double" ),
	new ValueFinfo< double >(
		"concInit", &MoleculeWrapper::getConcInit, 
		&MoleculeWrapper::setConcInit, "double" ),
	new ValueFinfo< double >(
		"volumeScale", &MoleculeWrapper::getVolumeScale, 
		&MoleculeWrapper::setVolumeScale, "double" ),
	new ValueFinfo< double >(
		"n", &MoleculeWrapper::getN, 
		&MoleculeWrapper::setN, "double" ),
	new ValueFinfo< double >(
		"conc", &MoleculeWrapper::getConc, 
		&MoleculeWrapper::setConc, "double" ),
	new ValueFinfo< int >(
		"mode", &MoleculeWrapper::getMode, 
		&MoleculeWrapper::setMode, "int" ),
	new ValueFinfo< int >(	// A backward compatibility hack
		"slaveEnable", &MoleculeWrapper::getSlaveEnable, 
		&MoleculeWrapper::setSlaveEnable, "int" ),
		// mode 0 is normal
		// mode 1 is sumtotalled. It is checked at reinit.
		// mode 2 is sumtotalled to conc. It is checked at reinit.
		// mode 3 or more is buffered.
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"reacOut", &MoleculeWrapper::getReacSrc, 
		"reinitIn, processIn", 1 ),
	new NSrc1Finfo< double >(
		"nOut", &MoleculeWrapper::getNSrc, 
		"reinitIn, processIn" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest2Finfo< double, double >(
		"reacIn", &MoleculeWrapper::reacFunc,
		&MoleculeWrapper::getReacConn, "", 1 ),
	new Dest2Finfo< double, double >(
		"prdIn", &MoleculeWrapper::prdFunc,
		&MoleculeWrapper::getPrdInConn, "" ),
	new Dest1Finfo< double >(
		"sumTotalIn", &MoleculeWrapper::sumTotalFunc,
		&MoleculeWrapper::getSumTotalInConn, "" ),
	new Dest1Finfo< ProcInfo >(
		"sumProcessIn", &MoleculeWrapper::sumProcessFunc,
		&MoleculeWrapper::getSumProcessInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &MoleculeWrapper::reinitFunc,
		&MoleculeWrapper::getProcessConn, "reacOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &MoleculeWrapper::processFunc,
		&MoleculeWrapper::getProcessConn, "reacOut", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &MoleculeWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"reac", &MoleculeWrapper::getReacConn,
		"reacOut, reacIn" ),
};

const Cinfo MoleculeWrapper::cinfo_(
	"Molecule",
	"Upinder S. Bhalla, 2005, NCBS",
	"Molecule: Pool of molecules.",
	"Neutral",
	MoleculeWrapper::fieldArray_,
	sizeof(MoleculeWrapper::fieldArray_)/sizeof(Finfo *),
	&MoleculeWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void MoleculeWrapper::reinitFuncLocal(  )
{
	A_ = B_ = total_ = 0.0;
	n_ = nInit_;
	if ( mode_ == 0 && sumTotalInConn_.nTargets() > 0 )
		mode_ = 1;
	else if ( (mode_ == 1 || mode_ == 1) && 	
		sumTotalInConn_.nTargets() == 0 )
		mode_ = 0;
	reacSrc_.send( n_ );
	nSrc_.send( n_ );
}

// Should do by func ptrs
void MoleculeWrapper::processFuncLocal( ProcInfo info )
{
	if ( mode_ == 0 ) {
		if ( n_ > EPSILON && B_ > EPSILON ) {
			double C = exp( -B_ * info->dt_ / n_ );
			n_ *= C + ( A_ / B_ ) * ( 1.0 - C );
		} else {
			n_ += ( A_ - B_ ) * info->dt_;
		}
		A_ = B_ = 0.0;
	} else if ( mode_ == 1 ) {
		// Hack to do sumtotals while we do not have a separate
		// process set up to do so. Roughly equivalent to 
		// old GENESIS version.
		n_ = total_;
		total_ = 0.0;
	} else if ( mode_ == 2 ) {
		// Hack to do sumtotals while we do not have a separate
		// process set up to do so. Roughly equivalent to 
		// old GENESIS version.
		n_ = total_ * volumeScale_;
		total_ = 0.0;
	} else { // buffering
		n_ = nInit_;
	}
	reacSrc_.send( n_ );
	nSrc_.send( n_ );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnMoleculeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MoleculeWrapper, processConn_ );
	return reinterpret_cast< MoleculeWrapper* >( ( unsigned long )c - OFFSET );
}

Element* sumProcessInConnMoleculeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MoleculeWrapper, sumProcessInConn_ );
	return reinterpret_cast< MoleculeWrapper* >( ( unsigned long )c - OFFSET );
}

