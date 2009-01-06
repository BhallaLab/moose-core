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
#include "DiskPanel.h"

/**
 * DiskPanel is derived from Panel, and almost everything in this class
 * is handled by the base class and a few virtual functions.
 */
const Cinfo* initDiskPanelCinfo()
{
	static string doc[] =
	{
		"Name", "DiskPanel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "DiskPanel: Disk panel shape for portion of compartmental surface.",
	};	
	static Cinfo diskPanelCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initPanelCinfo(),
		0,
		0,
		ValueFtype1< DiskPanel >::global()
	);

	return &diskPanelCinfo;
}

static const Cinfo* diskPanelCinfo = initDiskPanelCinfo();

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

