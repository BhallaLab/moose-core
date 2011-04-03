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
#include "ElementValueFinfo.h"
#include "LookupElementValueFinfo.h"
#include "Panel.h"

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////

static SrcFinfo0* toNeighbor()
{
	static SrcFinfo0 temp(
		"toNeighbor",
		"Identifies neighbors of the current panel"
	);
	return &temp;
}




const Cinfo* Panel::initCinfo()
{
	///////////////////////////////////////////////////////
	// Field definitions
	///////////////////////////////////////////////////////

	static ReadOnlyValueFinfo< Panel, unsigned int > nPts( 
		"nPts", 
		"Number of points used by panel to specify geometry",
		&Panel::getNpts
	);

	static ReadOnlyValueFinfo< Panel, unsigned int > nDims( 
		"nDims", 
		"Number of Dimensions used by panel to specify geometry",
		&Panel::getNdims
	);

	static ReadOnlyElementValueFinfo< Panel, unsigned int > numNeighbors( 
		"numNeighbors", 
		"Number of Neighbors of panel",
		&Panel::getNumNeighbors
	);

	static ReadOnlyValueFinfo< Panel, unsigned int > shapeId( 
		"shapeId", 
		"Identifier for shape type, as used by Smoldyn",
		&Panel::getShapeId
	);

	static LookupValueFinfo< Panel, unsigned int, double > x( 
		"x", 
		"x coordinate identified by index",
		&Panel::setX,
		&Panel::getX
	);
	static LookupValueFinfo< Panel, unsigned int, double > y( 
		"y", 
		"y coordinate identified by index",
		&Panel::setY,
		&Panel::getY
	);
	static LookupValueFinfo< Panel, unsigned int, double > z( 
		"z", 
		"z coordinate identified by index",
		&Panel::setZ,
		&Panel::getZ
	);

	static ValueFinfo< Panel, vector< double > > coords(
		"coords", 
		"All the coordinates for the panel. X vector, then Y, then Z"
		"Z can be left out for 2-D panels."
		"Z and Y can be left out for 1-D panels.",
		&Panel::setCoords,
		&Panel::getCoords
	);

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo neighbor(
		"neighbor",
		"Handles incoming message from neighbor",
		new OpFunc0< Panel >( &Panel::handleNeighbor ) 
	);

	///////////////////////////////////////////////////////
	// Put all the finfos together
	///////////////////////////////////////////////////////

	static Finfo* panelFinfos[] =
	{
		&nPts,				// ReadOnlyValue
		&nDims,				// ReadOnlyValue
		&numNeighbors,		// ReadOnlyValue
		&shapeId,			// ReadOnlyValue
		&x,					// LookupValue
		&y,					// LookupValue
		&z,					// LookupValue
		&coords,			// Value
		toNeighbor(),		// Src
		&neighbor			// Dest
	};

	static string doc[] =
	{
		"Name", "Panel",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "Panel: Base class for shapes. Also serves as interface to Smoldyn panel data struct.",
	};	
	static Cinfo panelCinfo(
		"Panel",
		Neutral::initCinfo(),
		panelFinfos,
		sizeof( panelFinfos )/sizeof(Finfo *),
		new Dinfo< Panel >()
	);

	return &panelCinfo;
}

static const Cinfo* panelCinfo = Panel::initCinfo();

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

unsigned int Panel::getNpts() const
{
	return this->localNpts();
}

// Default version
unsigned int Panel::localNpts() const {
	return 2;  // Centre coord, and then radius, with sign.
}

unsigned int Panel::getNdims() const
{
	return nDims_;
}

unsigned int Panel::getShapeId() const
{
	return this->localShapeId();
}

// Default version
unsigned int Panel::localShapeId() const
{
	return Moose::PSsph; 
}

unsigned int Panel::getNumNeighbors( const Eref& e, const Qinfo* q ) const
{
	return 0;
	/*
	return e.e->numTargets( neighborSrcFinfo->msg() ) + 
		e.e->numTargets( neighborDestFinfo->msg() );
		*/
}

void Panel::setCoords( vector< double > v )
{
	if ( v.size() == coords_.size() )
		coords_ = v;
	else
		cout << "Error: Panel::setCoord: argument vector dimension should be " << v.size() << ", was " << v.size() << endl;
}

vector< double > Panel::getCoords() const
{
	return coords_;
}

//////////////////////////////////////////////////////////
// Special Field function definitions to do with coords.
//////////////////////////////////////////////////////////

void Panel::localSetPos( unsigned int i, unsigned int dim, double value )
{
	assert( i * nDims_ + dim < coords_.size() );
	coords_[ i * nDims_ + dim ] = value;
}

void Panel::setX( unsigned int i, double value )
{
	this->localSetPos( i, 0, value );
}

void Panel::setY( unsigned int i, double value )
{
	this->localSetPos( i, 1, value );
}

void Panel::setZ( unsigned int i, double value )
{
	this->localSetPos( i, 2, value );
}

double Panel::localGetPos( unsigned int i, unsigned int dim ) const
{
	assert( i * nDims_ + dim < coords_.size() );
	return coords_[ i * nDims_ + dim ];
}

double Panel::getX( unsigned int i ) const
{
	return this->localGetPos( i, 0 );
}

double Panel::getY( unsigned int i ) const
{
	return this->localGetPos( i, 1 );
}

double Panel::getZ( unsigned int i ) const
{
	return this->localGetPos( i, 2 );
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
vector< double > Panel::getFiniteElementVertices( double area ) const
{
	vector< double > ret;
	this->localFiniteElementVertices( ret, area );
	return ret;
}

void Panel::localFiniteElementVertices( 
	vector< double >& ret, double area ) const
{
	// Fill up 'ret' here.
	;
}

/////////////////////////////////////////////////////////////////
void Panel::handleNeighbor()
{
	; // dummy func
}
