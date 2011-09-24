/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ElementValueFinfo.h"
#include "MeshEntry.h"
#include "Boundary.h"
#include "ChemMesh.h"

const Cinfo* MeshEntry::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyElementValueFinfo< MeshEntry, double > size(
			"size",
			"Volume of this MeshEntry",
			&MeshEntry::getSize
		);

		static ReadOnlyElementValueFinfo< MeshEntry, unsigned int > 
			dimensions (
			"dimensions",
			"number of dimensions of this MeshEntry",
			&MeshEntry::getDimensions
		);

		static ReadOnlyElementValueFinfo< MeshEntry, unsigned int > 
			meshType(
			"meshType",
		 	" The MeshType defines the shape of the mesh entry."
		 	" 0: Not assigned"
		 	" 1: cuboid"
		 	" 2: cylinder"
		 	" 3. cylindrical shell"
		 	" 4: cylindrical shell segment"
		 	" 5: sphere"
		 	" 6: spherical shell"
		 	" 7: spherical shell segment"
		 	" 8: Tetrahedral",
			&MeshEntry::getMeshType
		);

		static ReadOnlyElementValueFinfo< MeshEntry, vector< double >  >
			coordinates (
			"Coordinates",
			"Coordinates that define current MeshEntry. Depend on MeshType.",
			&MeshEntry::getCoordinates
		);

		static ReadOnlyElementValueFinfo< MeshEntry, vector< unsigned int >  >
			neighbors (
			"neighbors",
			"Indices of other MeshEntries that this one connects to",
			&MeshEntry::getNeighbors
		);

		static ReadOnlyElementValueFinfo< MeshEntry, vector< double >  >
			diffusionArea (
			"DiffusionArea",
			"Diffusion area for geometry of interface",
			&MeshEntry::getDiffusionArea
		);

		static ReadOnlyElementValueFinfo< MeshEntry, vector< double >  >
			diffusionScaling (
			"DiffusionScaling",
			"Diffusion scaling for geometry of interface",
			&MeshEntry::getDiffusionScaling
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< MeshEntry >( &MeshEntry::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< MeshEntry >( &MeshEntry::reinit ) );
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

		//////////////////////////////////////////////////////////////
		// SrcFinfo Definitions
		//////////////////////////////////////////////////////////////

	static Finfo* meshFinfos[] = {
		&size,			// Readonly Value
		&dimensions,	// Readonly Value
		&meshType,		// Readonly Value
		&coordinates,	// Readonly Value
		&neighbors,		// Readonly Value
		&diffusionArea,	// Readonly Value
		&diffusionScaling,	// Readonly Value
		&group,			// DestFinfo
		&proc,			// SharedFinfo
	};

	static Cinfo meshEntryCinfo (
		"MeshEntry",
		Neutral::initCinfo(),
		meshFinfos,
		sizeof( meshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< MeshEntry >()
	);

	return &meshEntryCinfo;
}

//////////////////////////////////////////////////////////////
// Class definitions
//////////////////////////////////////////////////////////////
static const Cinfo* poolCinfo = MeshEntry::initCinfo();

MeshEntry::MeshEntry()
	: parent_( 0 )
{;}

MeshEntry::MeshEntry( const ChemMesh* parent )
	: parent_( parent )
{;}

//////////////////////////////////////////////////////////////
// Process operations. Used for reac-diff calculations.
//////////////////////////////////////////////////////////////

/**
 * Update the diffusion rate terms. Note that these stay the same
 * through the entire clock tick, even if the GSL solver takes many
 * intermediate steps. The assumption is that the concs change slowly
 * enough that the diffusion terms are not hugely changed over the duration
 * of one tick. Also assume that diffusion itself is slow. The latter
 * assumption is OK for suitable grid sizes. The first assumption is OK
 * with a sensible tick step.
 */
void MeshEntry::process( const Eref& e, ProcPtr info )
{
	parent_->updateDiffusion( e.index().data() );
}

void MeshEntry::reinit( const Eref& e, ProcPtr info )
{
	;
}


//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double MeshEntry::getSize( const Eref& e, const Qinfo* q ) const
{
	return parent_->getMeshEntrySize( e.index().field() );
}

unsigned int MeshEntry::getDimensions( const Eref& e, const Qinfo* q ) const
{
	return parent_->getMeshDimensions( e.index().field() );
}

unsigned int MeshEntry::getMeshType( const Eref& e, const Qinfo* q ) const
{
	return parent_->getMeshType( e.index().field() );
}

vector< double >MeshEntry::getCoordinates( const Eref& e, const Qinfo* q ) const
{
	return parent_->getCoordinates( e.index().field() );
}

vector< unsigned int >MeshEntry::getNeighbors(
	const Eref& e, const Qinfo* q ) const
{
	return parent_->getNeighbors( e.index().field() );
}


vector< double >MeshEntry::getDiffusionArea( const Eref& e, const Qinfo* q ) const
{
	return parent_->getDiffusionArea( e.index().field() );
}


vector< double >MeshEntry::getDiffusionScaling( const Eref& e, const Qinfo* q ) const
{
	return parent_->getDiffusionScaling( e.index().field() );
}

