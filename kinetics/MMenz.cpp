

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "../basecode/header.h"
#include "lookupVolumeFromMesh.h"
// #include "FuncTerm.h"
#include "../basecode/SparseMatrix.h"
#include "../ksolve/KinSparseMatrix.h"
// #include "VoxelPoolsBase.h"
// #include "../mesh/VoxelJunction.h"
// #include "XferInfo.h"
// #include "KsolveBase.h"
// #include "Stoich.h"

#include "../ksolve/RateTerm.h"
class KsolveBase;
#include "../ksolve/Stoich.h"
#include "EnzBase.h"
#include "MMenz.h"
#define EPSILON 1e-15

const Cinfo* MMenz::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo setSolver( "setSolver",
			"Assigns solver to this MMEnz.",
			new EpFunc1< MMenz, ObjId >( &MMenz::setSolver ) );
		//////////////////////////////////////////////////////////////
		// Shared Msg Definitions
		//////////////////////////////////////////////////////////////
	static Finfo* mmEnzFinfos[] = {
		&setSolver,		// DestFinfo
	};

    static string doc[] = {
        "Name", "MMenz",
        "Author", "Upi Bhalla",
        "Description", "Class for MM (Michaelis-Menten) enzyme."
	};
	static Dinfo< MMenz > dinfo;
	static Cinfo MMenzCinfo (
		"MMenz",
		EnzBase::initCinfo(),
		mmEnzFinfos,
		sizeof( mmEnzFinfos ) / sizeof ( Finfo* ),
		&dinfo,
        doc,
        sizeof(doc)/sizeof(string)
	);

	return &MMenzCinfo;
}

//////////////////////////////////////////////////////////////

static const Cinfo* MMenzCinfo = MMenz::initCinfo();

static const SrcFinfo2< double, double >* subOut =
    dynamic_cast< const SrcFinfo2< double, double >* >(
	MMenzCinfo->findFinfo( "subOut" ) );

static const SrcFinfo2< double, double >* prdOut =
	dynamic_cast< const SrcFinfo2< double, double >* >(
	MMenzCinfo->findFinfo( "prdOut" ) );

//////////////////////////////////////////////////////////////
// MMenz internal functions
//////////////////////////////////////////////////////////////


MMenz::MMenz( )
	: 
		stoich_( 0 )
{;}

MMenz::~MMenz( )
{;}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void MMenz::vSetKm( const Eref& e, double v )
{
	Km_ = v;
	if ( stoich_ )
		stoich_->setMMenzKm( e, v );
}

void MMenz::vSetKcat( const Eref& e, double v )
{
	kcat_ = v;
	if ( stoich_ )
		stoich_->setMMenzKcat( e, v );
}

//////////////////////////////////////////////////////////////
// Utility function
//////////////////////////////////////////////////////////////

void MMenz::setSolver( const Eref& e, ObjId solver )
{
	static const DestFinfo* enzFinfo = dynamic_cast< const DestFinfo* >(
		EnzBase::initCinfo()->findFinfo( "enzDest" ) );
	static const SrcFinfo* subFinfo = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "subOut" ) );
	static const SrcFinfo* prdFinfo = dynamic_cast< const SrcFinfo* >(
		EnzBase::initCinfo()->findFinfo( "prdOut" ) );
	assert( enzFinfo );
	assert( subFinfo );
	assert( prdFinfo );

	assert( solver.element()->cinfo()->isA( "Stoich" ) );
	Stoich* stoichPtr = reinterpret_cast< Stoich* >( solver.data() );
	if ( stoich_ == stoichPtr )
		return;
	if (stoich_)
		stoich_->notifyRemoveMMenz( e );
	stoich_ = stoichPtr;

	/// Now set up the RateTerm
	vector< Id > enzvec;
	vector< Id > subvec;
	vector< Id > prdvec;
	unsigned int num = e.element()->getNeighbors( enzvec, enzFinfo );
	num = e.element()->getNeighbors( subvec, subFinfo );
	num = e.element()->getNeighbors( prdvec, prdFinfo );
	stoich_->installMMenz( e.id(), enzvec, subvec, prdvec );
	stoich_->setMMenzKm( e, Km_ );
	stoich_->setMMenzKcat( e, kcat_ );
}

