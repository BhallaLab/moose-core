/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "header.h"
#include "Panel.h"
#include "SpherePanel.h"

/**
 * HemispherePanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* SpherePanel::initCinfo()
{
	static string doc[] =
	{
		"Name", "SpherePanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SpherePanel: Spherical panel shape for portion of compartmental surface.",
	};	
	static Cinfo spherePanelCinfo(
		"SpherePanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< SpherePanel >()
	);

	return &spherePanelCinfo;
}

static const Cinfo* spherePanelCinfo = SpherePanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * The Sphere is defined by 3 pts: 
 * centre, radius[dim0], outward vec[dim0+ve]
 */
SpherePanel::SpherePanel( unsigned int nDims )
	: Panel( nDims, 2 )
{
		;
}

void SpherePanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

