/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#include "moose.h"
#include "Panel.h"
#include "SpherePanel.h"

/**
 * SpherePanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* initSpherePanelCinfo()
{
	static string doc[] =
	{
		"Name", "SpherePanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SpherePanel: Spherical panel shape for portion of compartmental surface.",
	};
	
	static Cinfo spherePanelCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initPanelCinfo(),
		0,
		0,
		ValueFtype1< SpherePanel >::global()
	);

	return &spherePanelCinfo;
}

static const Cinfo* spherePanelCinfo = initSpherePanelCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * Sphere is defined by 2 points: a centre point, and a radius at dim0.
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

