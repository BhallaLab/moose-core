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
#include "CylPanel.h"

/**
 * CylPanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* CylPanel::initCinfo()
{
	static string doc[] =
	{
		"Name", "CylPanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "CylPanel: Cylindrical panel shape for portion of compartmental surface.",
	};
	static Cinfo cylPanelCinfo(
		"CylPanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< CylPanel >()
	);

	return &cylPanelCinfo;
}

static const Cinfo* cylPanelCinfo = CylPanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * Cylinder is defined by 2 points for axis plus another for radius at dim0
 */
CylPanel::CylPanel( unsigned int nDims )
	: Panel( nDims, 3 ) // 3 points to define it.
{
		;
}

void CylPanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

