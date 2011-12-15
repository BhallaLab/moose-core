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
#include "ChemCompt.h"

static SrcFinfo1< double > *compartment() {
	static SrcFinfo1< double > compartment( 
			"compartment", 
			"Tracks all molecules in the compartment, also updates with size."
			);
	return &compartment;
}

static SrcFinfo0 *requestSize() {
	static SrcFinfo0 requestSize( 
			"requestSize", 
			"Requests size from geometry. "
			);
	return &requestSize;
}

const Cinfo* ChemCompt::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< ChemCompt, double > size(
			"size",
			"Size of compartment",
			&ChemCompt::setSize,
			&ChemCompt::getSize
		);

		static ValueFinfo< ChemCompt, unsigned int > dimensions(
			"dimensions",
			"Number of dimensions of this compartment. Usually 3 or 2",
			&ChemCompt::setDimensions,
			&ChemCompt::getDimensions
		);

		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////

		static DestFinfo group( "group",
			"Handle for grouping. Doesn't do anything.",
			new OpFuncDummy() );

		static DestFinfo handleSize( "handleSize",
			"Handle for size, part of SharedFinfo to connect to geometry.",
			new OpFunc1< ChemCompt, double >( &ChemCompt::setSize )
			);
		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

		static Finfo* geomShared[] = {
			requestSize(), &handleSize
		};

		static SharedFinfo geom( "geom",
			"Connects to Geometry tree(s) defining compt",
			geomShared, sizeof( geomShared ) / sizeof( const Finfo* )
		);

		//////////////////////////////////////////////////////////////
		// Field Element for the boundaries.
		// Assume about 16.
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< ChemCompt, Boundary > boundaryFinfo( 
			"boundary", 
			"Field Element for Boundaries",
			Boundary::initCinfo(),
			&ChemCompt::lookupBoundary,
			&ChemCompt::setNumBoundary,
			&ChemCompt::getNumBoundary,
			16
		);

	static Finfo* chemComptFinfos[] = {
		&size,	// Value
		&dimensions,	// Value
		requestSize(),	// SrcFinfo
		compartment(),	// SrcFinfo
		&group,			// DestFinfo
		&geom,			// SharedFinfo
		&boundaryFinfo,	// FieldElementFinfo
	};

	static Cinfo chemComptCinfo (
		"ChemCompt",
		Neutral::initCinfo(),
		chemComptFinfos,
		sizeof( chemComptFinfos ) / sizeof ( Finfo* ),
		new Dinfo< ChemCompt >()
	);

	return &chemComptCinfo;
}

//////////////////////////////////////////////////////////////
// Basic class Definitions
//////////////////////////////////////////////////////////////

static const Cinfo* chemComptCinfo = ChemCompt::initCinfo();

ChemCompt::ChemCompt()
	: size_( 1.0 ), dimensions_( 3 )
{
	;
}

//////////////////////////////////////////////////////////////
// MsgDest Definitions
//////////////////////////////////////////////////////////////

/*
bool ChemCompt::isInside( double x, double y, double z )
{
}
*/

void ChemCompt::extent( DataId di, double volume, double area, double perimeter )
{
	if ( dimensions_ == 3 ) 
		size_ = volume;
	else if ( dimensions_ == 2 ) 
		size_ = area;
	else if ( dimensions_ <= 1 ) 
		size_ = perimeter;
}

//////////////////////////////////////////////////////////////
// Field Definitions
//////////////////////////////////////////////////////////////

void ChemCompt::setSize( double v )
{
	size_ = v;
}

double ChemCompt::getSize() const
{
	return size_;
}

void ChemCompt::setDimensions( unsigned int v )
{
	if ( v <= 3 )
		dimensions_ = v;
	if ( dimensions_ == 0 )
		dimensions_ = 1;
}

unsigned int ChemCompt::getDimensions() const
{
	return dimensions_;
}

//////////////////////////////////////////////////////////////
// Element Field Definitions
//////////////////////////////////////////////////////////////

Boundary* ChemCompt::lookupBoundary( unsigned int index )
{
	if ( index < boundaries_.size() )
		return &( boundaries_[index] );
	cout << "Error: ChemCompt::lookupBoundary: Index " << index << 
		" >= vector size " << boundaries_.size() << endl;
	return 0;
}

void ChemCompt::setNumBoundary( unsigned int num )
{
	assert( num < 1000 ); // Pretty unlikely upper limit
	boundaries_.resize( num );
}

unsigned int ChemCompt::getNumBoundary() const
{
	return boundaries_.size();
}
