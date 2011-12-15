/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Boundary.h"
#include "MeshEntry.h"
#include "Stencil.h"
#include "ChemMesh.h"
#include "../ksolve/StoichHeaders.h"

// Not used anywhere currently. - Sid
static SrcFinfo0 groupSurfaces( 
		"groupSurfaces", 
		"Goes to all surfaces that define this ChemMesh"
);

const Cinfo* ChemMesh::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ReadOnlyValueFinfo< ChemMesh, double > size(
			"size",
			"Size of entire chemical domain",
			&ChemMesh::getEntireSize
		);

		static ReadOnlyValueFinfo< ChemMesh, unsigned int > dimensions(
			"dimensions",
			"Number of dimensions of this compartment. Usually 3 or 2",
			&ChemMesh::getDimensions
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo stoich( "stoich",
			"Handle Id of stoich. Used to set up connection from mesh to"
			"stoich for diffusion "
			"calculations. Somewhat messy way of doing it, should really "
			"use messaging.",
			new EpFunc1< ChemMesh, Id >( &ChemMesh::stoich )
		);

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		/*
		static Finfo* geomShared[] = {
			&requestSize, &handleSize
		};

		static SharedFinfo geom( "geom",
			"Connects to Geometry tree(s) defining compt",
			geomShared, sizeof( geomShared ) / sizeof( const Finfo* )
		);
		*/

		//////////////////////////////////////////////////////////////
		// Field Elements
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< ChemMesh, Boundary > boundaryFinfo( 
			"boundary", 
			"Field Element for Boundaries",
			Boundary::initCinfo(),
			&ChemMesh::lookupBoundary,
			&ChemMesh::setNumBoundary,
			&ChemMesh::getNumBoundary,
			16
		);

		static FieldElementFinfo< ChemMesh, MeshEntry > entryFinfo( 
			"mesh", 
			"Field Element for mesh entries",
			MeshEntry::initCinfo(),
			&ChemMesh::lookupEntry,
			&ChemMesh::setNumEntries,
			&ChemMesh::getNumEntries,
			1048576
		);

	static Finfo* chemMeshFinfos[] = {
		&size,			// ReadOnlyValue
		&dimensions,	// ReadOnlyValue
		&stoich,		// DestFinfo
		&entryFinfo,	// FieldElementFinfo
		&boundaryFinfo,	// Boundaries
	};

	static Cinfo chemMeshCinfo (
		"ChemMesh",
		Neutral::initCinfo(),
		chemMeshFinfos,
		sizeof( chemMeshFinfos ) / sizeof ( Finfo* ),
		new Dinfo< short >()
	);

	return &chemMeshCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemMeshCinfo = ChemMesh::initCinfo();

ChemMesh::ChemMesh()
	: 
		size_( 1.0 ),
		entry_( this )
{
	;
}

ChemMesh::~ChemMesh()
{ 
	for ( unsigned int i = 0; i < stencil_.size(); ++i ) {
		if ( stencil_[i] )
			delete stencil_[i];
	}
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////
void ChemMesh::stoich( const Eref& e, const Qinfo* q, Id stoichId )
{
	stoich_ = stoichId;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

double ChemMesh::getEntireSize() const
{
	return size_;
}

unsigned int ChemMesh::getDimensions() const
{
	return this->innerGetDimensions();
}

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

MeshEntry* ChemMesh::lookupEntry( unsigned int index )
{
	return &entry_;
}

void ChemMesh::setNumEntries( unsigned int num )
{
	this->innerSetNumEntries( num );
	// cout << "Warning: ChemMesh::setNumEntries: No effect. Use subclass-specific functions\nto build or resize mesh.\n";
}

unsigned int ChemMesh::getNumEntries() const
{
	return this->innerGetNumEntries();
}

//////////////////////////////////////////////////////////////
// Element Field Definitions for boundary
//////////////////////////////////////////////////////////////

Boundary* ChemMesh::lookupBoundary( unsigned int index )
{
	if ( index < boundaries_.size() )
		return &( boundaries_[index] );
	cout << "Error: ChemCompt::lookupBoundary: Index " << index << 
		" >= vector size " << boundaries_.size() << endl;
	return 0;
}

void ChemMesh::setNumBoundary( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	boundaries_.resize( num );
}

unsigned int ChemMesh::getNumBoundary() const
{
	return boundaries_.size();
}

//////////////////////////////////////////////////////////////
// Orchestrate diffusion calculations in Stoich. This simply updates
// the flux terms (du/dt due to diffusion). Virtual func, has to be
// defined for every Mesh class if it differs from below.
// Called from the MeshEntry.
//////////////////////////////////////////////////////////////

void ChemMesh::updateDiffusion( unsigned int meshIndex ) const
{
	// Later we'll have provision for multiple stoich targets.
	Stoich* s = reinterpret_cast< Stoich* >( stoich_.eref().data() );
	s->updateDiffusion( meshIndex, stencil_ );
}
