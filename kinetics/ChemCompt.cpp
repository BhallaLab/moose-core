/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ChemCompt.h"

static SrcFinfo1< double > size( 
		"size", 
		"Sends out # of molecules on each timestep"
);

const Cinfo* ChemCompt::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< ChemCompt, double > n(
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

		//////////////////////////////////////////////////////////////
		// SharedMsg Definitions
		//////////////////////////////////////////////////////////////

	static Finfo* chemComptFinfos[] = {
		&size,	// Value
		&dimensions,	// Value
		&group,			// DestFinfo
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

void ChemCompt::process( const ProcInfo* p, const Eref& e )
{
;
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
	dimensions_ = v;
}

unsigned int ChemCompt::getDimensions() const
{
	return dimensions_;
}
