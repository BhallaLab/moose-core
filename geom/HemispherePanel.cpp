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
#include "HemispherePanel.h"

/**
 * HemispherePanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* HemispherePanel::initCinfo()
{
	static string doc[] =
	{
		"Name", "HemispherePanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "HemispherePanel: Hemispherical panel shape for portion of compartmental surface.",
	};	
	static Cinfo hemispherePanelCinfo(
		"HemispherePanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< HemispherePanel >()
	);

	return &hemispherePanelCinfo;
}

static const Cinfo* hemispherePanelCinfo = HemispherePanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * The Hemisphere is defined by 3 pts: 
 * centre, radius[dim0], outward vec[dim0+ve]
 */
HemispherePanel::HemispherePanel( unsigned int nDims )
	: Panel( nDims, 3 )
{
		;
}

void HemispherePanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

