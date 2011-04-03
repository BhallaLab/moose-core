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
#include "RectPanel.h"

/**
 * RectPanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* RectPanel::initCinfo()
{
	static string doc[] =
	{
		"Name", "RectPanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "RectPanel: Rectangular panel shape for portion of compartmental surface.",
	};
	
	static Cinfo rectPanelCinfo(
		"RectPanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< RectPanel >()
	);

	return &rectPanelCinfo;
}

static const Cinfo* rectPanelCinfo = RectPanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * RectPanel is defined by 4 corners with CCW winding, followed by a
 * further 'front' vector whose meaning I need to check with Steven.
 */
RectPanel::RectPanel( unsigned int nDims )
	: Panel( nDims, 4 )
{
		;
}

void RectPanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

