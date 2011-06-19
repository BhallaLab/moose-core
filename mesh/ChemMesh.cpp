/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MeshEntry.h"
#include "ChemMesh.h"

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
		/*
		static FieldElementFinfo< ChemMesh, MeshEntry > boundaryFinfo( 
			"boundary", 
			"Field Element for Boundaries",
			Boundary::initCinfo(),
			&ChemMesh::lookupBoundary,
			&ChemMesh::setNumBoundary,
			&ChemMesh::getNumBoundary
		);
		*/

		static FieldElementFinfo< ChemMesh, MeshEntry > entryFinfo( 
			"meshEntries", 
			"Field Element for mesh entries",
			MeshEntry::initCinfo(),
			&ChemMesh::lookupEntry,
			&ChemMesh::setNumEntries,
			&ChemMesh::getNumEntries
		);

	static Finfo* chemComptFinfos[] = {
		&size,	// ReadOnlyValue
		&dimensions,	// ReadOnlyValue
		&entryFinfo,	// FieldElementFinfo
	};

	static Cinfo chemComptCinfo (
		"ChemMesh",
		Neutral::initCinfo(),
		chemComptFinfos,
		sizeof( chemComptFinfos ) / sizeof ( Finfo* ),
		new Dinfo< short >()
	);

	return &chemComptCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemComptCinfo = ChemMesh::initCinfo();

ChemMesh::ChemMesh()
	: 
		size_( 1.0 ),
		entry_( this )
{
	;
}

ChemMesh::~ChemMesh()
{ ; }

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////


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
	cout << "Warning: ChemMesh::setNumEntries: No effect. Use subclass-specific functions\nto build or resize mesh.\n";
}

unsigned int ChemMesh::getNumEntries() const
{
	return this->innerNumEntries();
}
