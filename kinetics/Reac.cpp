/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "../basecode/header.h"
#include "../ksolve/RateTerm.h"
#include "../basecode/SparseMatrix.h"
#include "../ksolve/KinSparseMatrix.h"
/*
#include "../ksolve/FuncTerm.h"
#include "../ksolve/VoxelPoolsBase.h"
#include "../mesh/VoxelJunction.h"
#include "../ksolve/XferInfo.h"
#include "../ksolve/KsolveBase.h"
class KinSparseMatrix;
*/
class KsolveBase;
#include "../ksolve/Stoich.h"
#include "../basecode/ElementValueFinfo.h"
#include "lookupVolumeFromMesh.h"
#include "Reac.h"

#define EPSILON 1e-15

static SrcFinfo2< double, double > *subOut() {
	static SrcFinfo2< double, double > subOut(
			"subOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &subOut;
}

static SrcFinfo2< double, double > *prdOut() {
	static SrcFinfo2< double, double > prdOut(
			"prdOut",
			"Sends out increment of molecules on product each timestep"
			);
	return &prdOut;
}

const Cinfo* Reac::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ElementValueFinfo< Reac, double > kf(
			"numKf",
			"Forward rate constant, in # units",
			&Reac::setNumKf,
			&Reac::getNumKf
		);

		static ElementValueFinfo< Reac, double > kb(
			"numKb",
			"Reverse rate constant, in # units",
			&Reac::setNumKb,
			&Reac::getNumKb
		);

		static ElementValueFinfo< Reac, double > Kf(
			"Kf",
			"Forward rate constant, in concentration units",
			&Reac::setConcKf,
			&Reac::getConcKf
		);

		static ElementValueFinfo< Reac, double > Kb(
			"Kb",
			"Reverse rate constant, in concentration units",
			&Reac::setConcKb,
			&Reac::getConcKb
		);

		static ReadOnlyElementValueFinfo< Reac, unsigned int > numSub(
			"numSubstrates",
			"Number of substrates of reaction",
			&Reac::getNumSub
		);

		static ReadOnlyElementValueFinfo< Reac, unsigned int > numPrd(
			"numProducts",
			"Number of products of reaction",
			&Reac::getNumPrd
		);
    	static ReadOnlyElementValueFinfo< Reac, ObjId > compartment(
        	"compartment",
        	"ObjId of parent compartment of Reac. "
        	"If the compartment isn't"
        	"available this returns the root ObjId.",
        	&Reac::getCompartment
    	);
	
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		/*
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< Reac >( &Reac::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< Reac >( &Reac::reinit ) );
			*/
		static DestFinfo setSolver( "setSolver",
			"Assigns solver to this Reac.",
			new EpFunc1< Reac, ObjId >( &Reac::setSolver ) );

		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo subDest( "subDest",
				"Handles # of molecules of substrate",
				new OpFunc1< Reac, double >( &Reac::sub ) );
		static DestFinfo prdDest( "prdDest",
				"Handles # of molecules of product",
				new OpFunc1< Reac, double >( &Reac::prd ) );
		static Finfo* subShared[] = {
			subOut(), &subDest
		};
		static Finfo* prdShared[] = {
			prdOut(), &prdDest
		};
		static SharedFinfo sub( "sub",
			"Connects to substrate pool",
			subShared, sizeof( subShared ) / sizeof( const Finfo* )
		);
		static SharedFinfo prd( "prd",
			"Connects to substrate pool",
			prdShared, sizeof( prdShared ) / sizeof( const Finfo* )
		);
		/*
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);
		*/


	static Finfo* reacFinfos[] = {
		&kf,	// Value
		&kb,	// Value
		&Kf,	// Value
		&Kb,	// Value
		&numSub,	// ReadOnlyValue
		&numPrd,	// ReadOnlyValue
		&compartment,	// ReadOnlyValue
		&sub,				// SharedFinfo
		&prd,				// SharedFinfo
		// &proc,				// SharedFinfo
		&setSolver,			// DestFinfo
	};

	static string doc[] =
	{
		"Name", "Reac",
		"Author", "Upinder S. Bhalla, 2012, 2020 NCBS",
		"Description", "Class for reactions. Handles both standalone and"
		"solved configurations"
	};

    static Dinfo<Reac> dinfo;
	static Cinfo reacCinfo (
		"Reac",
		Neutral::initCinfo(),
		reacFinfos,
		sizeof( reacFinfos ) / sizeof ( Finfo* ),
		&dinfo,
		doc,
		sizeof( doc ) / sizeof( string )
	);

	return &reacCinfo;
}

 static const Cinfo* reacCinfo = Reac::initCinfo();

//////////////////////////////////////////////////////////////
// Reac internal functions
//////////////////////////////////////////////////////////////

Reac::Reac( )
	: concKf_( 0.1 ), concKb_( 0.2 ), stoich_( 0 )
{ ; }

Reac::~Reac( )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Reac::setNumKf( const Eref& e, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, subOut(), 0 );
	concKf_ = v * volScale;
	if ( stoich_ )
		stoich_->setReacKf( e, concKf_ );
}

double Reac::getNumKf( const Eref& e ) const
{
	// Return value for voxel 0. Conceivably I might want to use the
	// DataId part to specify which voxel to use, but that isn't in the
	// current definition for Reacs as being a single entity for the
	// entire compartment.
	double volScale = convertConcToNumRateUsingMesh( e, subOut(), 0 );
	return concKf_ / volScale;
}

void Reac::setNumKb( const Eref& e, double v )
{
	double volScale = convertConcToNumRateUsingMesh( e, prdOut(), 0 );
	concKb_ = v * volScale;
	if ( stoich_ )
		stoich_->setReacKb( e, concKb_ );
}

double Reac::getNumKb( const Eref& e ) const
{
	double volScale = convertConcToNumRateUsingMesh( e, prdOut(), 0 );
	return concKb_ / volScale;
}

void Reac::setConcKf( const Eref& e, double v )
{
	concKf_ = v;
	if ( stoich_ )
		stoich_->setReacKf( e, v );
}

double Reac::getConcKf( const Eref& e ) const
{
	return concKf_;
}

void Reac::setConcKb( const Eref& e, double v )
{
	concKb_ = v;
	if ( stoich_ )
		stoich_->setReacKb( e, v );
}

double Reac::getConcKb( const Eref& e ) const
{
	return concKb_;
}

unsigned int Reac::getNumSub( const Eref& e ) const
{
	const vector< MsgFuncBinding >* mfb =
		e.element()->getMsgAndFunc( subOut()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

unsigned int Reac::getNumPrd( const Eref& e ) const
{
	const vector< MsgFuncBinding >* mfb =
		e.element()->getMsgAndFunc( prdOut()->getBindIndex() );
	assert( mfb );
	return ( mfb->size() );
}

ObjId Reac::getCompartment( const Eref& e ) const
{
    return getCompt( e.id() );
}


//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Reac::sub(double v)
{;}

void Reac::prd(double v)
{;}

//////////////////////////////////////////////////////////////
// Assign a new solver to the reac.
//////////////////////////////////////////////////////////////

void Reac::setSolver( const Eref& e, ObjId newStoich )
{
	if ( newStoich.bad() ) {
		cout << "Warning: Reac::setSolver: Bad Stoich " << 
				e.id().path() << endl;
		return;
	}
	if ( newStoich == ObjId() ) { // Unsetting stoich.
		if ( stoich_ != 0 )
			stoich_->notifyRemoveReac( e );
		stoich_ = 0;
		return;
	}
	if ( !newStoich.element()->cinfo()->isA( "Stoich" ) ) {
		cout << "Warning: Reac::setSolver: object " << newStoich.path() << "is not a Stoich for " << 
				e.id().path() << endl;
		return;
	}

	assert( newStoich.element()->cinfo()->isA( "Stoich" ) );
	Stoich* stoichPtr = reinterpret_cast< Stoich* >( newStoich.eref().data( ) );
	if ( stoich_ == stoichPtr )
		return;

	if ( stoich_ != 0 )
		stoich_->notifyRemoveReac( e );
	
	stoich_ = stoichPtr;
	/// Some of this should go into notifyAddReac
	vector< Id > sub;
	vector< Id > prd;
	e.element()->getNeighbors( sub, subOut() );
	e.element()->getNeighbors( prd, prdOut() );

	stoich_->installReaction( e.id(), sub, prd );
	stoich_->setReacKf( e, concKf_ );
	stoich_->setReacKb( e, concKb_ );
}

