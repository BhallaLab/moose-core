/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include <math.h>
#include "Molecule.h"

const double Molecule::EPSILON = 1.0e-15;

const Cinfo* initMoleculeCinfo()
{
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &Molecule::reinitFunc ) ),
	};
	static TypeFuncPair reacTypes[] =
	{
		TypeFuncPair( Ftype2< double, double >::global(),
			RFCAST( &Molecule::reacFunc ) ),
		TypeFuncPair( Ftype1< double >::global(), 0 )
	};

	static Finfo* moleculeFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nInit", 
			ValueFtype1< double >::global(),
			GFCAST( &MoleculeWrapper::getNInit ), 
			RFCAST( &MoleculeWrapper::setNInit ) 
		),
		new ValueFinfo( "volumeScale", 
			ValueFtype1< double >::global(),
			GFCAST( &MoleculeWrapper::getVolumeScale ), 
			RFCAST( &MoleculeWrapper::setVolumeScale )
		),
		new ValueFinfo( "n", 
			ValueFtype1< double >::global(),
			GFCAST( &MoleculeWrapper::getN ), 
			RFCAST( &MoleculeWrapper::setN )
		),
		new ValueFinfo( "mode", 
			ValueFtype1< int >::global(),
			GFCAST( &MoleculeWrapper::getMode ), 
			RFCAST( &MoleculeWrapper::setMode )
		),
		new ValueFinfo( "slave_enable", 
			ValueFtype1< int >::global(),
			GFCAST( &MoleculeWrapper::getMode ), 
			RFCAST( &MoleculeWrapper::setMode )
		),
		new ValueFinfo( "conc", 
			ValueFtype1< double >::global(),
			GFCAST( &MoleculeWrapper::getConc ), 
			RFCAST( &MoleculeWrapper::setConc )
		),
		new ValueFinfo( "concInit", 
			ValueFtype1< double >::global(),
			GFCAST( &MoleculeWrapper::getConcInit ), 
			RFCAST( &MoleculeWrapper::setConcInit )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "nSrc", Ftype1< double >::global ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	
		new DestFinfo( "prd"
			Ftype2< double, double >::global(),
			RFCAST( &MoleculeWrapper::prdFunc )
		),
	
		new DestFinfo( "sumTotal"
			Ftype1< double >::global(),
			RFCAST( &MoleculeWrapper::sumTotalFunc )
		),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processTypes, 2 ),
		new SharedFinfo( "reac", reacTypes, 2 ),
	};

	static Cinfo moleculeCinfo(
		"Molecule",
		"Upinder S. Bhalla, 2005, NCBS",
		"Molecule: Pool of molecules.",
		initNeutralCinfo(),
		moleculeFinfos,
		sizeof( moleculeFinfos )/sizeof(Finfo *),
		ValueFtype1< Molecule >::global()
	);

	return &moleculeCinfo;
};

static const Cinfo* moleculeCinfo = initMoleculeCinfo();

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
