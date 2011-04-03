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
#include "DiskPanel.h"

/**
 * DiskPanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* DiskPanel::initCinfo()
{
	static string doc[] =
	{
		"Name", "DiskPanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "DiskPanel: Disk panel shape for portion of compartmental surface.",
	};	
	static Cinfo diskPanelCinfo(
		"DiskPanel",
		Panel::initCinfo(),
		0,
		0,
		new Dinfo< DiskPanel >()
	);

	return &diskPanelCinfo;
}

static const Cinfo* diskPanelCinfo = DiskPanel::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * The three points that define a disk are: centre, radius[dim0], normal.
 */
DiskPanel::DiskPanel( unsigned int nDims )
	: Panel( nDims, 3 )
{
		;
}

void DiskPanel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

