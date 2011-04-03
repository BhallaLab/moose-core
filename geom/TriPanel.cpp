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
#include "TriPanel.h"

/**
 * TriPanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* TriPanel::initCinfo()
{
	static string doc[] =
	{
		"Name","TriPanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "TriPanel: Triangular panel shape for portion of compartmental surface.",
	};
	
	static Cinfo triPanelCinfo(
		"TriPanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< TriPanel >()
	);

	return &triPanelCinfo;
}

static const Cinfo* triPanelCinfo = TriPanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * Triangle is defined by 3 points with CCW winding.
 */
TriPanel::TriPanel( unsigned int nDims )
	: Panel( nDims, 3 )
{
		;
}

void TriPanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

