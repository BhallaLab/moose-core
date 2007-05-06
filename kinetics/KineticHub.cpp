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
#include "RateTerm.h"
#include "KineticHub.h"
#include "Molecule.h"
#include "Reaction.h"
#include "Enzyme.h"

const Cinfo* initKineticHubCinfo()
{
	static TypeFuncPair processTypes[] =
	{
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::processFunc ) ),
		TypeFuncPair( Ftype1< ProcInfo >::global(),
			RFCAST( &KineticHub::reinitFunc ) ),
	};

	static Finfo* kineticHubFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "nMol", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNmol ), 
			&dummyFunc
		),
		new ValueFinfo( "nReac", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNreac ), 
			&dummyFunc
		),
		new ValueFinfo( "nEnz", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KineticHub::getNenz ), 
			&dummyFunc
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "process", processTypes, 2 ),
		/*
		new SolveFinfo( "molSolve", molFields, 
			sizeof( molFields ) / sizeof( const Finfo* ) );
			*/
	};

	static Cinfo kineticHubCinfo(
		"KineticHub",
		"Upinder S. Bhalla, 2007, NCBS",
		"KineticHub: Object for controlling reaction systems on behalf of the\nStoich object. Interfaces both with the reaction system\n(molecules, reactions, enzymes\nand user defined rate terms) and also with the Stoich\nclass which generates the stoichiometry matrix and \nhandles the derivative calculations.",
		initNeutralCinfo(),
		kineticHubFinfos,
		sizeof(kineticHubFinfos )/sizeof(Finfo *),
		ValueFtype1< KineticHub >::global()
	);

	return &kineticHubCinfo;
}

static const Cinfo* kineticHubCinfo = initKineticHubCinfo();

/*
static const unsigned int reacSlot =
	initMoleculeCinfo()->getSlotIndex( "reac" );
static const unsigned int nSlot =
	initMoleculeCinfo()->getSlotIndex( "nSrc" );
*/


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

unsigned int KineticHub::getNmol( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

unsigned int KineticHub::getNreac( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

unsigned int KineticHub::getNenz( const Element* e )
{
	return static_cast< KineticHub* >( e->data() )->nMol_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void KineticHub::reinitFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

void KineticHub::processFunc( const Conn& c, ProcInfo info )
{
	// Element* e = c.targetElement();
	// static_cast< KineticHub* >( e->data() )->processFuncLocal( e, info );
}

/*
void KineticHubWrapper::molSizesFuncLocal( int nMol, int nBuf, int nSumTot )
{
			vector< unsigned long > segments( 1, 0 );
			nMol_ = segments[0] = nMol;
			Field molSolve( this, "molSolve" );
			molSolve->resize( this, segments );
			nBuf_ = segments[0] = nBuf;
			Field bufSolve( this, "bufSolve" );
			bufSolve->resize( this, segments );
			nSumTot_ = segments[0] = nSumTot;
			Field sumTotSolve( this, "sumTotSolve" );
			sumTotSolve->resize( this, segments );
}
void KineticHubWrapper::rateSizesFuncLocal( int nReac, int nEnz, int nMmEnz )
{
			vector< unsigned long > segments( 1, 0 );
			segments[0] = nReac;
			Field reacSolve( this, "reacSolve" );
			reacSolve->resize( this, segments );
			segments[0] = nEnz;
			Field enzSolve( this, "enzSolve" );
			enzSolve->resize( this, segments );
			segments[0] = nMmEnz;
			Field mmEnzSolve( this, "mmEnzSolve" );
			mmEnzSolve->resize( this, segments );
}
void KineticHubWrapper::molConnectionsFuncLocal( vector< double >*  S, vector< double >*  Sinit, vector< Element *>*  elist )
{
			if ( nMol_ + nBuf_ + nSumTot_ != elist->size() ) {
				cerr << "Error: KineticHub::molConnections: Number of molecules does not match elist size\n";
				return;
			}
			Field molSolve( this, "molSolve" );
			Field sumTotSolve( this, "sumTotSolve" );
			Field bufSolve( this, "bufSolve" );
			S_ = S;
			Sinit_ = Sinit;
			unsigned long i;
			unsigned long max;
			for ( i = 0; i < nMol_; i++ )
				zombify( (*elist)[i], molSolve );
			max = nMol_ + nSumTot_;
			for ( ; i < max; i++ )
				zombify( (*elist)[i], sumTotSolve );
			max = nMol_ + nSumTot_ + nBuf_;
			for ( ; i < max; i++ )
				zombify( (*elist)[i], bufSolve );
}
void KineticHubWrapper::reacConnectionFuncLocal( int rateTermIndex, Element* reac )
{
			Field reacSolve( this, "reacSolve" );
			zombify( reac, reacSolve );
			reacIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::enzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field enzSolve( this, "enzSolve" );
			zombify( enz, enzSolve );
			enzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::mmEnzConnectionFuncLocal( int rateTermIndex, Element* enz )
{
			Field mmEnzSolve( this, "mmEnzSolve" );
			zombify( enz, mmEnzSolve );
			mmEnzIndex_.push_back( rateTermIndex );
}
void KineticHubWrapper::molFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				molSrc_.sendTo( index, (*S_)[index] );
			} else if ( mode == SOLVER_SET ) {
				(*S_)[index] = n;
				(*Sinit_)[index] = nInit;
			} else if ( mode == SOLVER_REBUILD ) {
				rebuildFlag_ = 1;
			}
}
void KineticHubWrapper::bufFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
			} else if ( mode == SOLVER_SET ) {
				(*S_)[ index + nMol_ + nSumTot_ ] = 
					(*Sinit_)[ index + nMol_ + nSumTot_ ] = nInit;
			}
}
void KineticHubWrapper::sumTotFuncLocal( double n, double nInit, int mode, long index )
{
			if ( mode == SOLVER_GET ) {
				sumTotSrc_.sendTo( index, (*S_)[index + nMol_ ] );
			}
}
void KineticHubWrapper::reacFuncLocal( double kf, double kb, long index )
{
			unsigned long i = reacIndex_[ index ];
			if ( i >= 0 && i < rates_->size() - useOneWayReacs_ ) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( kf, 0 );
					( *rates_ )[i + 1]->setRates( kb, 0 );
				} else {
					( *rates_ )[i]->setRates( kf, kb );
				}
			}
}
void KineticHubWrapper::enzFuncLocal( double k1, double k2, double k3, long index )
{
			unsigned int i = enzIndex_[ index ];
			if ( i < rates_->size() - useOneWayReacs_ - 1) {
				if ( useOneWayReacs_ ) {
					( *rates_ )[i]->setRates( k1, 0 );
					( *rates_ )[i + 1]->setRates( k2, 0 );
					( *rates_ )[i + 2]->setRates( k3, 0 );
				} else {
					( *rates_ )[i]->setRates( k1, k2 );
					( *rates_ )[i + 1]->setRates( k3, 0 );
				}
			}
}
void KineticHubWrapper::mmEnzFuncLocal( double k1, double k2, double k3, long index )
{
			double Km = ( k2 + k3 ) / k1 ;
			unsigned int i = mmEnzIndex_[ index ];
			if ( i >= 0 && i < rates_->size() )
				( *rates_ )[i]->setRates( Km, k3 );
}
*/

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
	/*
void KineticHub::zombify( Element* e, const Finfo* solveFinfo )
{
	Field f( e, "process" );
	if ( !f.dropAll() ) {
		cerr << "Error: Failed to delete process message into " <<
			e->path() << "\n";
	}
	Field ms( e, "solve" );
	if ( !solveSrc.add( ms ) ) {
		cerr << "Error: Failed to add solve message from solver " <<
			solveSrc.path() << " to zombie " << e->path() << "\n";
	}
}
	*/
