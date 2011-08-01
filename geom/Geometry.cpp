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
#include "Geometry.h"

/**
 * The Geometry corresponds to the Smoldyn surfacesuperstruct.
 * It manages multiple surfaces that matter to a given solver.
 * It mostly handles a list of surfaces, but has a couple of global control
 * parameters for tolerances, and messages to send these on to the
 * solver when updated.
 */
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > returnSize(
	"returnSize",
	"Return size of compartment"
);
const Cinfo* Geometry::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	static ValueFinfo< Geometry, double > epsilon( 
		"epsilon", 
		"epsilon is the max deviation of surface-point from surface."
		"I think it refers to when the molecule is "
		"stuck to the surface. Need to check with Steven.",
		&Geometry::setEpsilon,
		&Geometry::getEpsilon
	);

	static ValueFinfo< Geometry, double > neighdist( 
		"neighdist", 
		"neighdist is capture distance from one panel to another."
		"When a molecule diffuses off one panel and is "
		"within neighdist of the other, it is captured by the second.",
		&Geometry::setNeighDist,
		&Geometry::getNeighDist
	);
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo handleSizeRequest( 
		"handleSizeRequest",
		"Handles a request for size. Part of SharedMsg to ChemCompt.",
		new EpFunc0< Geometry >( &Geometry::handleSizeRequest )
	);
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	static Finfo* comptShared[] = {
		&handleSizeRequest,
		&returnSize
	};
	
	static SharedFinfo compt( "compt",
		"Connects to compartment(s) to specify geometry.",
		comptShared, sizeof( comptShared ) / sizeof( const Finfo* ) 
	);

	///////////////////////////////////////////////////////
	static Finfo* geometryFinfos[] =
	{
		&epsilon,	// value
		&neighdist, // value
		&compt		// SharedFinfo
	};
	static string doc[] =
	{
		"Name", "Geometry",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Geometry: Interface object between Smoldyn (by Steven Andrews) and MOOSE, corresponding to the Smoldyn surfacesuperstruct",
	};

	static Cinfo geometryCinfo(
		"Geometry",
		Neutral::initCinfo(),
		geometryFinfos,
		sizeof( geometryFinfos )/sizeof(Finfo *),
		new Dinfo< Geometry >()
	);

	return &geometryCinfo;
}

static const Cinfo* geometryCinfo = Geometry::initCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Geometry::Geometry()
{
		;
}

///////////////////////////////////////////////////
// Field access functions.
///////////////////////////////////////////////////
		
void Geometry::setEpsilon( double value )
{
	if ( value >= 0.0 )
		epsilon_ = value;
}

double Geometry::getEpsilon() const
{
	return epsilon_;
}

void Geometry::setNeighDist( double value )
{
	if ( value >= 0.0 )
		neighDist_ = value;
}

double Geometry::getNeighDist() const
{
	return neighDist_;
}

///////////////////////////////////////////////////
// Dest function.
///////////////////////////////////////////////////

void Geometry::handleSizeRequest( const Eref& e, const Qinfo* q )
{
	returnSize.send( e, q->threadNum(), 0.0 );
}

