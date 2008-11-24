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

const Cinfo* initPanelCinfo()
{
	/*
	static Finfo* reacShared[] =
	{
		new DestFinfo( "reac", Ftype2< double, double >::global(),
			RFCAST( &Panel::reacFunc ) ),
		new SrcFinfo( "n", Ftype1< double >::global() )
	};
	*/

	static Finfo* panelFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////
		new LookupFinfo( "x", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Panel::getX ), 
			RFCAST( &Panel::setX ) 
		),
		new LookupFinfo( "y", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Panel::getY ), 
			RFCAST( &Panel::setY ) 
		),
		new LookupFinfo( "z", 
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &Panel::getZ ), 
			RFCAST( &Panel::setZ ) 
		),

		new ValueFinfo( "nPts", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Panel::getNpts ), 
			&dummyFunc
		),
		new ValueFinfo( "nDims", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Panel::getNdims ), 
			&dummyFunc
		),
		new ValueFinfo( "nNeighbors", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Panel::getNneighbors ), 
			&dummyFunc
		),
		new ValueFinfo( "shapeId", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &Panel::getShapeId ), 
			&dummyFunc
		),
		new ValueFinfo( "coords", 
			ValueFtype1< vector< double > >::global(),
			GFCAST( &Panel::getCoords ), 
			&dummyFunc,
			"Coords is used for getting all the shape position info in one vector. The # and meaning "
			"of each coord depends on the shape subtype. The coord vector is accessed as "
			"coord[index * nDim + dim]"
		),
	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "neighborSrc", Ftype0::global() ),
	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "neighbor", Ftype0::global(),
			&dummyFunc ),
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
	};

	static string doc[] =
	{
		"Name", "Panel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Panel: Base class for shapes. Also serves as interface to Smoldyn panel data struct.",
	};	
	static Cinfo panelCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		panelFinfos,
		sizeof( panelFinfos )/sizeof(Finfo *),
		ValueFtype1< Panel >::global()
	);

	return &panelCinfo;
}

static const Cinfo* panelCinfo = initPanelCinfo();
static const Finfo* neighborSrcFinfo = initPanelCinfo()->findFinfo( "neighborSrc" );
static const Finfo* neighborDestFinfo = initPanelCinfo()->findFinfo( "neighbor" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

/**
 * Panel is defined by dimensions, currently always 3, and nPts.
 * The points sets up the coordinates of the panel as well as the 'outward'
 * direction.
 */
Panel::Panel( unsigned int nDims, unsigned int nPts )
	: nDims_( nDims ), coords_( ( nPts + 1 ) * nDims, 0.0 )
{
		;
}

///////////////////////////////////////////////////
// Regular field functions 
///////////////////////////////////////////////////

unsigned int Panel::getNpts( Eref e )
{
	return static_cast< Panel* >( e.data() )->localNpts();
}

// Default version
unsigned int Panel::localNpts() const {
	return 2;  // Centre coord, and then radius, with sign.
}

unsigned int Panel::getNdims( Eref e )
{
	return static_cast< Panel* >( e.data() )->nDims_;
}

unsigned int Panel::getShapeId( Eref e )
{
	return static_cast< Panel* >( e.data() )->localShapeId();
}

// Default version
unsigned int Panel::localShapeId() const {
	return Moose::PSsph; 
}

unsigned int Panel::getNneighbors( Eref e )
{
	return e.e->numTargets( neighborSrcFinfo->msg() ) + 
		e.e->numTargets( neighborDestFinfo->msg() );
}

vector< double > Panel::getCoords( Eref e )
{
	return static_cast< Panel* >( e.data() )->coords_;
}

//////////////////////////////////////////////////////////
// Special Field function definitions to do with coords.
//////////////////////////////////////////////////////////

// We don't have dynamic shapes yet, so this update to the solver is
// not needed. Furthermore we don't want to tie this to Smoldyn.
void Panel::setPos( const Conn* c, double value, 
	unsigned int i, unsigned int dim )
{
	static_cast< Panel* >( c->data() )->localSetPos( value, i, dim );
}

void Panel::localSetPos( double value, unsigned int i, unsigned int dim )
{
	assert( i * nDims_ + dim < coords_.size() );
	coords_[ i * nDims_ + dim ] = value;
}

void Panel::setX( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 0 );
}

void Panel::setY( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 1 );
}

void Panel::setZ( const Conn* c, double value, const unsigned int& i )
{
	setPos( c, value, i, 2 );
}

double Panel::getPos( Eref e, unsigned int i, unsigned int dim)
{
	return static_cast< Panel* >( e.data() )->localGetPos( i, dim );
}

double Panel::localGetPos( unsigned int i, unsigned int dim )
{
	assert( i * nDims_ + dim < coords_.size() );
	return coords_[ i * nDims_ + dim ];
}

double Panel::getX( Eref e, const unsigned int& i )
{
	return getPos( e, i, 0 );
}

double Panel::getY( Eref e, const unsigned int& i )
{
	return getPos( e, i, 1 );
}

double Panel::getZ( Eref e, const unsigned int& i )
{
	return getPos( e, i, 2 );
}

///////////////////////////////////////////////////
// Finite element mesh generator for surface
///////////////////////////////////////////////////
/**
 * For future use: Convert the current shape into a set of 
 * triangular
 * finite element vertices (assuming 3 d). The fineness of the
 * grid is set by the 'area' argument.
 */
vector< double > Panel::getFiniteElementVertices(
		Eref e, double area )
{
	vector< double > ret;
	static_cast< Panel* >( e.data() )->
		localFiniteElementVertices( ret, area );
	return ret;
}

void Panel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}
