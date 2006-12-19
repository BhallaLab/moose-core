/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include <math.h>
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
		"volumeScale", &MoleculeWrapper::getVolumeScale, 
		&MoleculeWrapper::setVolumeScale, "double" ),
	new ValueFinfo< double >(
		"n", &MoleculeWrapper::getN, 
		&MoleculeWrapper::setN, "double" ),
	new ValueFinfo< int >(
		"mode", &MoleculeWrapper::getMode, 
		&MoleculeWrapper::setMode, "int" ),
	new ValueFinfo< int >(
		"slaveEnable", &MoleculeWrapper::getMode, 
		&MoleculeWrapper::setMode, "int" ),
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"conc", &MoleculeWrapper::getConc, 
		&MoleculeWrapper::setConc, "double" ),
	new ValueFinfo< double >(
		"concInit", &MoleculeWrapper::getConcInit, 
		&MoleculeWrapper::setConcInit, "double" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"reacOut", &MoleculeWrapper::getReacSrc, 
		"reinitIn, processIn", 1 ),
	new NSrc1Finfo< double >(
		"nOut", &MoleculeWrapper::getNSrc, 
		"reinitIn, processIn" ),
	new SingleSrc3Finfo< double, double, int >(
		"solveOut", &MoleculeWrapper::getSolveSrc, 
		"", 1 ),
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
		&MoleculeWrapper::getProcessConn, "reacOut, nOut", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &MoleculeWrapper::processFunc,
		&MoleculeWrapper::getProcessConn, "reacOut, nOut", 1 ),
	new Dest1Finfo< double >(
		"solveIn", &MoleculeWrapper::solveFunc,
		&MoleculeWrapper::getProcessConn, "", 1 ),
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
		"solve", &MoleculeWrapper::getProcessConn,
		"processIn, reinitIn, solveIn, solveOut" ),
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
// EvalField function definitions
///////////////////////////////////////////////////

double MoleculeWrapper::localGetConc() const
{
			if ( volumeScale_ > 0.0 )
				return n_ / volumeScale_ ;
			else
				return n_;
}
void MoleculeWrapper::localSetConc( double value ) {
			if ( volumeScale_ > 0.0 )
				n_ = value * volumeScale_ ;
			else
				n_ = value;
}
double MoleculeWrapper::localGetConcInit() const
{
			if ( volumeScale_ > 0.0 )
				return nInit_ / volumeScale_ ;
			else
				return nInit_;
}
void MoleculeWrapper::localSetConcInit( double value ) {
			if ( volumeScale_ > 0.0 )
				nInit_ = value * volumeScale_ ;
			else
				nInit_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void MoleculeWrapper::reinitFuncLocal(  )
{
			A_ = B_ = total_ = 0.0;
			n_ = nInit_;
			if ( mode_ == 0 && sumTotalInConn_.nTargets() > 0 )
				mode_ = 1;
			else if ( mode_ == 1 && sumTotalInConn_.nTargets() == 0 )
				mode_ = 0;
			reacSrc_.send( n_ );
			nSrc_.send( n_ );
}
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
				n_ = total_;
				total_ = 0.0;
			} else if ( mode_ == 2 ) {
				n_ = total_ * volumeScale_;
				total_ = 0.0;
			} else { 
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

/*
Element* solveConnMoleculeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MoleculeWrapper, solveConn_ );
	return reinterpret_cast< MoleculeWrapper* >( ( unsigned long )c - OFFSET );
}
*/

Element* sumProcessInConnMoleculeLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MoleculeWrapper, sumProcessInConn_ );
	return reinterpret_cast< MoleculeWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
bool MoleculeWrapper::isSolved() const
{
	return ( solveSrc_.targetFunc(0) && 
		solveSrc_.targetFunc(0) != dummyFunc0 );
}
void MoleculeWrapper::solverUpdate( const Finfo* f, SolverOp s ) const
{
	if ( solveSrc_.targetFunc(0) && 
		solveSrc_.targetFunc(0) != dummyFunc0 ) {
		if ( s == SOLVER_SET ) {
			if ( f->name() == "n" || f->name() == "nInit" ||
				f->name() == "conc" || f->name() == "concInit" )
				solveSrc_.send( n_, nInit_, SOLVER_SET );
		} else if ( s == SOLVER_GET ) {
			if ( f->name() == "n" || f->name() == "conc" )
				solveSrc_.send( n_, nInit_, SOLVER_GET );
		}
	}
}
