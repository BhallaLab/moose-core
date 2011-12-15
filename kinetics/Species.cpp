/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Species.h"

static SrcFinfo1< double > *sendMolWt() {
	static SrcFinfo1< double > sendMolWt( 
			"sendMolWt", 
			"returns molWt."
			);
	return &sendMolWt;
}

const Cinfo* Species::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< Species, double > molWt(
			"molWt",
			"Molecular weight of species",
			&Species::setMolWt,
			&Species::getMolWt
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo handleMolWtRequest( "handleMolWtRequest",
			"Handle requests for molWt.",
			new EpFunc0< Species >( &Species::handleMolWtRequest )
			);
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		static Finfo* poolShared[] = {
			&handleMolWtRequest, sendMolWt()
		};

		static SharedFinfo pool( "pool",
			"Connects to pools of this Species type",
			poolShared, sizeof( poolShared ) / sizeof( const Finfo* )
		);

	static Finfo* speciesFinfos[] = {
		&molWt,			// Value
		&pool,			// SharedFinfo, connecting to any pool of this Species
	};

	static Cinfo speciesCinfo (
		"Species",
		Neutral::initCinfo(),
		speciesFinfos,
		sizeof( speciesFinfos ) / sizeof ( Finfo* ),
		new Dinfo< Species >()
	);

	return &speciesCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* speciesCinfo = Species::initCinfo();

Species::Species()
	: molWt_( 1 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

void Species::handleMolWtRequest( const Eref& e, const Qinfo* q )
{
	sendMolWt()->send( e, q->threadNum(), molWt_ );
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void Species::setMolWt( double v )
{
	molWt_ = v;
}

double Species::getMolWt() const
{
	return molWt_;
}

