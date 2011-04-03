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
#include "Surface.h"

const Cinfo* Surface::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< Surface, double > volume( 
			"volume", 
			"This is something I'll need to write a function to compute."
			"Perhaps have an update routine "
			"as it may be hard to compute but is needed often by the "
			"molecules.",
			&Surface::getVolume
		);
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
	static SrcFinfo0 absorb( 
		"absorb",
		"these help the system define non-standard operations for what a molecule does when it hits "
		"a surface.The default is reflect.As a molecule may interact with multiple surfaces, it "
		"isn't enough to confer a property on the molecule itself. We have to use messages. "
		"Perhaps we don't need these, but instead put entities on the surface which the molecule "
		"interacts with if it doesn't do the basic reflect operation." );
	static SrcFinfo0 transmit( 
			"transmit",
			"Surface lets molecules through" );
	static SrcFinfo0 jump( "jump", "dunno" );
	static SrcFinfo0 mixture( "mixture", "dunno" );
	static SrcFinfo3< double, double, double > surface( 
			"surface", 
			"Connects up to a compartment, either as interior or exterior Args are volume, area, perimeter" );
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////

	static Finfo* surfaceFinfos[] =
	{
		&volume,		// ReadOnly Value
		&absorb,		// Src
		&transmit,		// Src
		&jump,		// Src
		&mixture,		// Src
		&surface,		// Src
	};

	static Cinfo surfaceCinfo(
		"Surface",
		Neutral::initCinfo(),
		surfaceFinfos,
		sizeof( surfaceFinfos )/sizeof(Finfo *),
		new Dinfo< Surface >
	);

	return &surfaceCinfo;
}

static const Cinfo* surfaceCinfo = Surface::initCinfo();

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

double Surface::getVolume() const
{
	return volume_;
}
