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
#include "KinCompt.h"

/**
 * The KinCompt is a compartment for kinetic calculations. It doesn't
 * really correspond to a single Smoldyn concept, but it encapsulates
 * many of them into the traditional compartmental view. It connects up
 * with one or more surfaces which collectively define its volume and
 * kinCompt.
 * It also maps onto the SBML concept for compartments. It permits
 * the formation of compartments without surfaces but this is 
 * discouraged.
 */
const Cinfo* initKinComptCinfo()
{
	// Connects to molecules, which trigger a request for volume info.
	// Volume here may be size?
	static Finfo* extentShared[] =
	{
		// args are size, nDimensions
		new SrcFinfo( "returnExtent", 
			Ftype2< double, unsigned int >::global() ),
		new DestFinfo( "requestExtent", Ftype0::global(),
			RFCAST( &KinCompt::requestExtent ) ),
	};

	static Finfo* kinComptFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		// Volume is the compartment volume. If there are no
		// messages to surfaces this returns the local value.
		// The setVolume only works if there are no surfaces anywhere.
		// Otherwise the surfaces override it
		new ValueFinfo( "volume", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getVolume ), 
			RFCAST( &KinCompt::setVolume )
		),
		new ValueFinfo( "area", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getArea ), 
			RFCAST( &KinCompt::setArea )
		),
		new ValueFinfo( "perimeter", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getPerimeter ), 
			RFCAST( &KinCompt::setPerimeter )
		),

		// This takes whichever of the above is appropriate for the # of
		// dimensions. Means the same thing as the SBML size.
		new ValueFinfo( "size", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getSize ), 
			RFCAST( &KinCompt::setSize )
		),

		new ValueFinfo( "numDimensions", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KinCompt::getNumDimensions ), 
			RFCAST( &KinCompt::setNumDimensions )
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		// This goes to the compartment that encloses the current one.
		// Appropriate even for 2d enclosed by 3d and so on.
		new SrcFinfo( "outside", Ftype0::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		// This handles the 'outside' message from interior compartments.
		new DestFinfo( "inside", Ftype0::global(),
			&dummyFunc ),

		/**
	 	* Gets input from a single exterior surface
	 	*/
		new DestFinfo( "exterior", 
			Ftype3< double, double, double >::global(),
			RFCAST( &KinCompt::exteriorFunction ) ),
			
		/**
	 	* Gets input from possibly many interior surfaces
	 	*/
		new DestFinfo( "interior", 
			Ftype3< double, double, double >::global(),
			RFCAST( &KinCompt::interiorFunction ) ),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "extent", extentShared,
			sizeof( extentShared ) / sizeof( Finfo* ) ),
	};

	static Cinfo kinComptCinfo(
		"KinCompt",
		"Upinder S. Bhalla, 2007, NCBS",
		"KinCompt: Kinetic compartment. Has its on volume, or gets +ve or negative contributions from varous surface objects",
		initNeutralCinfo(),
		kinComptFinfos,
		sizeof( kinComptFinfos )/sizeof(Finfo *),
		ValueFtype1< KinCompt >::global()
	);

	return &kinComptCinfo;
}

static const Cinfo* kinComptCinfo = initKinComptCinfo();

static const unsigned int extentSlot =
	initKinComptCinfo()->getSlotIndex( "extent.returnExtent" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

KinCompt::KinCompt()
	: size_( 1.0 ), volume_( 1.0 ), area_( 1.0 ), perimeter_( 1.0 ), numDimensions_( 3 )
{
		;
}

///////////////////////////////////////////////////
// Field access functions.
///////////////////////////////////////////////////
		
void KinCompt::setVolume( const Conn& c, double value )
{
	if ( value < 0.0 )
		return;
	static_cast< KinCompt* >( c.data() )->innerSetVolume( value );
}

void KinCompt::innerSetVolume( double value )
{
	volume_ = value;
	if ( numDimensions_ == 3 )
		size_ = value;
}

double KinCompt::getVolume( const Element* e )
{
	return static_cast< KinCompt* >( e->data() )->volume_;
}

void KinCompt::setArea( const Conn& c, double value )
{
	if ( value < 0.0 )
		return;
	static_cast< KinCompt* >( c.data() )->innerSetArea( value );
}

void KinCompt::innerSetArea( double value )
{
	area_ = value;
	if ( numDimensions_ == 2 )
		size_ = value;
}

double KinCompt::getArea( const Element* e )
{
	return static_cast< KinCompt* >( e->data() )->volume_;
}


void KinCompt::setPerimeter( const Conn& c, double value )
{
	if ( value < 0.0 )
		return;
	static_cast< KinCompt* >( c.data() )->innerSetPerimeter( value );
}

void KinCompt::innerSetPerimeter( double value )
{
	perimeter_ = value;
	if ( numDimensions_ == 1 )
		size_ = value;
}

double KinCompt::getPerimeter( const Element* e )
{
	return static_cast< KinCompt* >( e->data() )->perimeter_;
}


void KinCompt::setSize( const Conn& c, double value )
{
	if ( value < 0.0 )
		return;
	static_cast< KinCompt* >( c.data() )->innerSetSize( value );
}

void KinCompt::innerSetSize( double value )
{
	size_ = value;
	if ( numDimensions_ == 3 )
		volume_ = value;
	else if ( numDimensions_ == 2 )
		area_ = value;
	else if ( numDimensions_ == 1 )
		perimeter_ = value;
}

double KinCompt::getSize( const Element* e )
{
	return static_cast< KinCompt* >( e->data() )->size_;
}

void KinCompt::setNumDimensions( const Conn& c, unsigned int value )
{
	if ( value == 0 || value > 3 )
		return;
	static_cast< KinCompt* >( c.data() )->numDimensions_ = value;
}

unsigned int KinCompt::getNumDimensions( const Element* e )
{
	return static_cast< KinCompt* >( e->data() )->numDimensions_;
}


///////////////////////////////////////////////////
// MsgDest functions.
///////////////////////////////////////////////////

void KinCompt::requestExtent( const Conn& c )
{
	static_cast< KinCompt* >( c.data() )->
		innerRequestExtent( c.targetElement() );
}

void KinCompt::innerRequestExtent( const Element* e ) const 
{
	send2< double, unsigned int >( e, extentSlot, size_, numDimensions_ );
}

void KinCompt::localExteriorFunction( double v1, double v2, double v3 )
{
	volume_ = v1;
	area_ = v2;
	perimeter_ = v3;
	if ( numDimensions_ == 3 )
		size_ = volume_;
	else if ( numDimensions_ == 2 )
		size_ = area_;
	else if ( numDimensions_ == 1 )
		size_ = perimeter_;
}

void KinCompt::exteriorFunction( 
	const Conn& c, double v1, double v2, double v3 )
{
	static_cast< KinCompt* >( c.data() )->localExteriorFunction( v1, v2, v3 );
}

void KinCompt::localInteriorFunction( double v1, double v2, double v3 )
{
	if ( numDimensions_ == 3 ) {
		volume_ -= v1;
		area_ += v2;
		size_ = volume_;
	} else if ( numDimensions_ == 2 ) {
		area_ -= v2;
		size_ = area_;
	} else if ( numDimensions_ == 1 ) {
		perimeter_ -= v3;
		size_ = perimeter_;
	}
}

void KinCompt::interiorFunction( 
	const Conn& c, double v1, double v2, double v3 )
{
	static_cast< KinCompt* >( c.data() )->localInteriorFunction( v1, v2, v3 );
}
