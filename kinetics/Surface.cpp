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
#include "Surface.h"

const Cinfo* initSurfaceCinfo()
{
	static Finfo* surfaceFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "volume", 
			ValueFtype1< double >::global(),
			GFCAST( &Surface::getVolume ), 
			&dummyFunc,
			"This is something I'll need to write a function to compute.Perhaps have an update routine "
			"as it may be hard to compute but is needed often by the molecules."
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "absorb", Ftype0::global(),
			"these help the system define non-standard operations for what a molecule does when it hits "
			"a surface.The default is reflect.As a molecule may interact with multiple surfaces, it "
			"isn't enough to confer a property on the molecule itself. We have to use messages. "
			"Perhaps we don't need these, but instead put entities on the surface which the molecule "
			"interacts with if it doesn't do the basic reflect operation." ),
		new SrcFinfo( "transmit", Ftype0::global() ),
		new SrcFinfo( "jump", Ftype0::global() ),
		new SrcFinfo( "mixture", Ftype0::global() ),
		new SrcFinfo( "surface", 
			Ftype3< double, double, double >::global(),
			"Connects up to a compartment, either as interior or exterior Args are volume, area, perimeter" ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	};

	static Cinfo surfaceCinfo(
		"Surface",
		"Upinder S. Bhalla, 2007, NCBS",
		"Surface: Interface object between Smoldyn (by Steven Andrews) and MOOSE.",
		initNeutralCinfo(),
		surfaceFinfos,
		sizeof( surfaceFinfos )/sizeof(Finfo *),
		ValueFtype1< Surface >::global()
	);

	return &surfaceCinfo;
}

static const Cinfo* surfaceCinfo = initSurfaceCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

Surface::Surface()
{
		;
}

///////////////////////////////////////////////////
// Field access functions.
///////////////////////////////////////////////////

double Surface::getVolume( const Element* e )
{
	return static_cast< Surface* >( e->data() )->volume_;
}
