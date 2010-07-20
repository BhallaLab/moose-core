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
#include "../ksolve/KineticManager.h"
#include "Molecule.h"
#include "Reaction.h"
#include "Enzyme.h"

// void rescaleTree( Eref e, double ratio );

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
	static Finfo* extentShared[] =
	{
		new SrcFinfo( "returnExtent", 
			Ftype2< double, unsigned int >::global(),
			"args are size, nDimensions" ),
		new DestFinfo( "requestExtent", Ftype0::global(),
			RFCAST( &KinCompt::requestExtent ) ),
	};

	static Finfo* kinComptFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new ValueFinfo( "volume", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getVolume ), 
			RFCAST( &KinCompt::setVolume ),
			"Volume is the compartment volume. If there are no messages to surfaces this returns the local value.\n"
			"The setVolume only works if there are no surfaces anywhere.Otherwise the surfaces override it \n"
			"Assigning any of the following four parameters causes automatic rescaling of rates throughout the model."
		),
		new ValueFinfo( "area", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getArea ), 
			RFCAST( &KinCompt::setArea ),
			"Area of compartment"
		),
		new ValueFinfo( "perimeter", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getPerimeter ), 
			RFCAST( &KinCompt::setPerimeter ),
			"Perimeter of compartment"
		),
		new ValueFinfo( "size", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getSize ), 
			RFCAST( &KinCompt::setSize ),
			"This is equal to whichever of volume, area, or perimeter "
			"is appropriate for the "
			"# of dimensions. Means the same thing as the "
			"SBML size."
		),
		new ValueFinfo( "numDimensions", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &KinCompt::getNumDimensions ), 
			RFCAST( &KinCompt::setNumDimensions ),
			"Number of dimensions of this compartment. Typically 3, "
			"but for a membrane compartment we might have 2."
		),
		new ValueFinfo( "x", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getX ), 
			RFCAST( &KinCompt::setX ),
			"X coordinate: for display."
		),
		new ValueFinfo( "y", 
			ValueFtype1< double >::global(),
			GFCAST( &KinCompt::getY ), 
			RFCAST( &KinCompt::setY ),
			"Y coordinate: for display."
		),

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "outside", Ftype0::global(),
			"This goes to the compartment that encloses the current "
			"one.Appropriate even for 2d enclosed by 3d and so on." ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "inside", Ftype0::global(),
			&dummyFunc,
			"This handles the 'outside' message from interior compartments." ),
		new DestFinfo( "exterior", 
			Ftype3< double, double, double >::global(),
			RFCAST( &KinCompt::exteriorFunction ),
			"Gets input from a single exterior surface" ),
		new DestFinfo( "interior", 
			Ftype3< double, double, double >::global(),
			RFCAST( &KinCompt::interiorFunction ),
			"Gets input from possibly many interior surfaces" ),
		new DestFinfo( "rescaleSize", 
			Ftype1< double >::global(),
			RFCAST( &KinCompt::rescaleFunction ),
			"Rescales the volume by the specified ratio. NewVol = ratio * old." ),
		new DestFinfo( "sizeWithoutRescale", 
			Ftype1< double >::global(),
			RFCAST( &KinCompt::setSizeWithoutRescale ),
			"Assigns size without rescaling the entire model." ),
		new DestFinfo( "volumeFromChild", 
			Ftype2< string, double >::global(),
			RFCAST( &KinCompt::setVolumeFromChild ),
			"Assigns volume based on request from child Molecule."
			"Applies the following logic:\n"
			"	- If first assignment: Assign without rescaling"
			"	- If later assignment, same vol: Keep tally, silently"
			"	- If later assignment, new vol: Complain, tally"
			"	- If later new vols outnumber original vol:"
			"Complain louder."
			),
	///////////////////////////////////////////////////////
	// Synapse definitions
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "extent", extentShared,
			sizeof( extentShared ) / sizeof( Finfo* ),
			"Connects to molecules, which trigger a request for volume info. Volume here may be size?" ),
	};

	static string doc[] =
	{
		"Name", "KinCompt",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description",
 		"The KinCompt is a compartment for kinetic calculations. "
		"It doesn't really correspond to a single Smoldyn concept, "
		"but it encapsulates many of them into the traditional "
		"compartmental view. It connects up with one or more "
		"surfaces which collectively define its volume and "
		"area.\n "
 		"It also maps onto the SBML concept for compartments. "
		"It permits the formation of compartments without surfaces "
		"but this is discouraged."
	};

	static Cinfo kinComptCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		kinComptFinfos,
		sizeof( kinComptFinfos )/sizeof(Finfo *),
		ValueFtype1< KinCompt >::global()
	);

	return &kinComptCinfo;
}

static const Cinfo* kinComptCinfo = initKinComptCinfo();

static const Slot extentSlot =
	initKinComptCinfo()->getSlot( "extent.returnExtent" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

KinCompt::KinCompt()
	: 	size_( 1.0 / ( Molecule::NA * 1e-3 ) ), 
		volume_( 1.0 / ( Molecule::NA * 1e-3 ) ), 
		area_( 1.0 ), 
		perimeter_( 1.0 ), 
		numDimensions_( 3 ),
		numAssigned_( 0 ),
		numMatching_( 0 )
{
		;
}

///////////////////////////////////////////////////
// Field access functions.
///////////////////////////////////////////////////
		
void KinCompt::setVolume( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->innerSetSize( 
		c->target(), value );
}

double KinCompt::getVolume( Eref e )
{
	KinCompt* kc = static_cast< KinCompt* >( e.data() );
	if ( kc->numDimensions_  == 3 )
		return kc->size_;
	return kc->volume_;
}

void KinCompt::setArea( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->innerSetSize(
		c->target(), value );
}

double KinCompt::getArea( Eref e )
{
	KinCompt* kc = static_cast< KinCompt* >( e.data() );
	if ( kc->numDimensions_  == 2 )
		return kc->size_;
	return kc->area_;
}

void KinCompt::setPerimeter( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->innerSetSize(
		c->target(), value );
}

double KinCompt::getPerimeter( Eref e )
{
	KinCompt* kc = static_cast< KinCompt* >( e.data() );
	if ( kc->numDimensions_  == 1 )
		return kc->size_;
	return kc->perimeter_;
}

void KinCompt::setSize( const Conn* c, double value )
{
	if ( value <= 0.0 ) {
		cout << "Error: KinCompt::setSize: value = " << value << 
			", must be positive\n";
		return;
	}
	static_cast< KinCompt* >( c->data() )->innerSetSize( 
		c->target(), value );
}

void KinCompt::setSizeWithoutRescale( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->innerSetSize( 
		c->target(), value, 1 );
}

void KinCompt::innerSetSize( Eref e, double value, bool ignoreRescale )
{
	assert( size_ > 0.0 );
	double ratio = value/size_;
	size_ = value;
	if ( numDimensions_ == 3 )
		volume_ = value;
	else if ( numDimensions_ == 2 )
		area_ = value;
	else if ( numDimensions_ == 1 )
		perimeter_ = value;

	// Here we scan through all children telling them to rescale.
	if ( !ignoreRescale )
		rescaleTree( e, ratio );
}

double KinCompt::getSize( Eref e )
{
	return static_cast< KinCompt* >( e.data() )->size_;
}

void KinCompt::setNumDimensions( const Conn* c, unsigned int value )
{
	if ( value == 0 || value > 3 )
		return;
	static_cast< KinCompt* >( c->data() )->numDimensions_ = value;
}

unsigned int KinCompt::getNumDimensions( Eref e )
{
	return static_cast< KinCompt* >( e.data() )->numDimensions_;
}
void KinCompt::setX( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->x_ = value;
}

double KinCompt::getX( Eref e )
{
	return static_cast< KinCompt* >( e.data() )->x_;
}

void KinCompt::setY( const Conn* c, double value )
{
	static_cast< KinCompt* >( c->data() )->y_ = value;
}

double KinCompt::getY( Eref e )
{
	return static_cast< KinCompt* >( e.data() )->y_;
}

///////////////////////////////////////////////////
// MsgDest functions.
///////////////////////////////////////////////////

void KinCompt::requestExtent( const Conn* c )
{
	static_cast< KinCompt* >( c->data() )->
		innerRequestExtent( c->target() );
}

void KinCompt::innerRequestExtent( Eref e ) const 
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
	const Conn* c, double v1, double v2, double v3 )
{
	static_cast< KinCompt* >( c->data() )->localExteriorFunction( v1, v2, v3 );
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
	const Conn* c, double v1, double v2, double v3 )
{
	static_cast< KinCompt* >( c->data() )->localInteriorFunction( v1, v2, v3 );
}

void KinCompt::rescaleFunction( const Conn* c, double ratio )
{
	double size = static_cast< KinCompt* >( c->data() )->size_;
	assert( size > 0.0 );
	static_cast< KinCompt* >( c->data() )->innerSetSize(
		c->target(), size * ratio );
}

void KinCompt::setVolumeFromChild( const Conn* c, string ch, double v )
{
	static_cast< KinCompt* >( c->data() )->innerSetVolumeFromChild( 
		c->target(), ch, v );
}

void KinCompt::innerSetVolumeFromChild( Eref pa, string ch, double v )
{
	// cout << "setting vol of " << pa.id().path() << " to " << v << " from child " << ch << ", ass:mat = " << numAssigned_ << ":" << numMatching_ << endl;
	if ( numAssigned_ == 0 ) {
		size_ = v;
		++numMatching_;
	} else if ( fabs( 1.0 - size_ / v ) < 1.0e-3 ) {
		++numMatching_;
	} else {
		cout << "Warning: KinCompt::innerSetVolumeFromChild: on" <<
		pa.id().path() << "\nOrig vol != new vol: ( " << size_ << 
		" != " << v << " ) on child " << ch << endl;
		if (  numMatching_ * 2 < numAssigned_ ) {
			cout << "\nNote: new vol is used by less than half the children\n";
		}
	}
	++numAssigned_;
}

///////////////////////////////////////////////////
// Utility functions for volume and size management.
///////////////////////////////////////////////////

Eref getNearestKinCompt( Eref e )
{
	static const Finfo* parentFinfo = 
		initNeutralCinfo()->findFinfo( "parent" );
	Id pa;
	bool ret = get< Id >( e, parentFinfo, pa );
	assert( ret );
	while( pa != Id() ) {
		Eref e = pa.eref();
		if ( e.e->cinfo()->isA( initKinComptCinfo() ) )
			return e;
		ret = get< Id >( e, parentFinfo, pa );
		assert( ret );
	}
	return Eref::root(); // Failed
}

/**
 * This is an extern function used by molecules, enzymes and reacs.
 * Traverses toward the root till it finds a KinCompt to get volScale.
 * If it runs into / return 1
 * If it runs into a KineticManager (happens with legacy kkit simulations)
 * uses the volScale on the kinetic manager.
 */

double getVolScale( Eref e )
{
	static const Finfo* sizeFinfo = 
		initKinComptCinfo()->findFinfo( "size" );

	Eref kc = getNearestKinCompt( e );
	if ( !( kc == Eref::root() ) ) {
		assert( kc.e->cinfo()->isA( initKinComptCinfo() ) );
		double size = 1.0;
		bool ret = get< double >( kc, sizeFinfo, size );
		assert( ret );
		assert( size > 0.0 );
		// Here we need to put in units too.
		//return 6e20 * size;
		return ( Molecule::NA * 1e-3 ) * size;
	}
	cout << "KinCompt.cpp:getVolScale: Failed to find KinCompt for volume\n";
	return 1.0;
}

void setParentalVolScale( Eref e, double volScale )
{
	/*
	static const Finfo* volFromChildFinfo = 
		initKinComptCinfo()->findFinfo( "volumeFromChild" );
		*/

	Eref kc = getNearestKinCompt( e );
	if ( !( kc == Eref::root() ) ) {
		if ( volScale <= 0.0 ) {
			cout << "Error: setParentalVolScale on " << 
				e.id().path() << 
				": volScale should be > 0: " << volScale << endl;
			return;
		}
		//double vol = volScale / 6e20;
		double vol = volScale / ( Molecule::NA * 1e-3 );
		//assert( volFromChildFinfo != 0 );
		bool ret = set< string, double >( 
			kc, "volumeFromChild", e.id().path(), vol);
		assert( ret );
		return;
	}
	cout << "KinCompt.cpp:setParentalVolScale: Failed to find KinCompt for volume\n";
}

/**
 * Recursively goes through all children, rescaling volumes.
 * Does NOT rescale current Eref.
 */
void rescaleTree( Eref e, double ratio )
{
	static const Finfo* childListFinfo = 
		initNeutralCinfo()->findFinfo( "childList" );
	static const Finfo* rescaleMolFinfo = 
		initMoleculeCinfo()->findFinfo( "rescaleVolume" );
	static const Finfo* rescaleReacFinfo = 
		initReactionCinfo()->findFinfo( "rescaleRates" );
	static const Finfo* rescaleEnzFinfo = 
		initEnzymeCinfo()->findFinfo( "rescaleRates" );
	static const Finfo* rescaleKinComptFinfo = 
		initKinComptCinfo()->findFinfo( "rescaleSize" );

	assert( childListFinfo != 0 && rescaleMolFinfo != 0 && rescaleReacFinfo != 0 && rescaleEnzFinfo != 0 && rescaleKinComptFinfo != 0 );

	vector< Id > kids;
	get< vector< Id > >( e, childListFinfo, kids );

	for( vector< Id >::iterator i = kids.begin(); i != kids.end(); ++i ) {
		if ( i->eref().e->cinfo()->isA( initReactionCinfo() ) )
			set< double >( i->eref(), rescaleReacFinfo, ratio );
		else if ( i->eref().e->cinfo()->isA( initEnzymeCinfo() ) )
			set< double >( i->eref(), rescaleEnzFinfo, ratio );
		else if ( i->eref().e->cinfo()->isA( initMoleculeCinfo() ) )
			set< double >( i->eref(), rescaleMolFinfo, ratio );

		// This sets off its own rescale recursive command.
		if ( i->eref().e->cinfo()->isA( initKinComptCinfo() ) )
			set< double >( i->eref(), rescaleKinComptFinfo, ratio );
		else 
			rescaleTree( i->eref(), ratio );
	}
}
