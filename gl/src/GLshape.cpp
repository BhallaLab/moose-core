/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "shell/Shell.h"

#include "GLshape.h"

const Cinfo* initGLshapeCinfo()
{
	static Finfo* processShared[] = 
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
			       RFCAST( &GLshape::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
			       RFCAST( &GLshape::reinitFunc ) ),
	};
	static Finfo* process = new SharedFinfo( "process", processShared,
						 sizeof( processShared ) / sizeof( Finfo* ),
						 "shared message to receive Process messages from scheduler objects");

	static Finfo* GLshapeFinfos[] = 
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "color",
				ValueFtype1< double >::global(),
				GFCAST( &GLshape::getColor ),
				RFCAST( &GLshape::setColor )
				),
		new ValueFinfo( "xoffset",
				ValueFtype1< double >::global(),
				GFCAST( &GLshape::getXOffset ),
				RFCAST( &GLshape::setXOffset )
				),
		new ValueFinfo( "yoffset",
				ValueFtype1< double >::global(),
				GFCAST( &GLshape::getYOffset ),
				RFCAST( &GLshape::setYOffset )
				),
		new ValueFinfo( "zoffset",
				ValueFtype1< double >::global(),
				GFCAST( &GLshape::getZOffset ),
				RFCAST( &GLshape::setZOffset )
				),
		new ValueFinfo( "len",
				ValueFtype1< double >::global(),
				GFCAST( &GLshape::getLen ),
				RFCAST( &GLshape::setLen )
				),
		new ValueFinfo( "shapetype",
				ValueFtype1< int >::global(),
				GFCAST( &GLshape::getShapeType ),
				RFCAST( &GLshape::setShapeType )
				),
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		//		parser,
		process,
	};

	// Schedule molecules for the slower clock, stage 0.
	static SchedInfo schedInfo[] = { { process, 0, 0 } };
	
	static string doc[] = 
	{
		"Name", "GLshape",
		"Author", "Karan Vasudeva, 2009, NCBS",
		"Description", "GLshape: the equivalent of xshape for use with GLview",
	};
	
	static Cinfo glshapeCinfo(
				 doc,
				 sizeof( doc ) / sizeof( string ),
				 initNeutralCinfo(),
				 GLshapeFinfos,
				 sizeof( GLshapeFinfos ) / sizeof( Finfo * ),
				 ValueFtype1< GLshape >::global(),
				 schedInfo, 1
				 );

	return &glshapeCinfo;
}

static const Cinfo* glshapeCinfo = initGLshapeCinfo();

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

GLshape::GLshape()
	:
	color_( 0.0 ),
	xoffset_( 0.0 ),
	yoffset_( 0.0 ),
	zoffset_( 0.0 ),
	len_( 0.0 ),
	shapetype_( CUBE )
{
}

GLshape::~GLshape()
{
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void GLshape::setColor( const Conn* c, double color )
{
	static_cast< GLshape * >( c->data() )->innerSetColor( color );
}

void GLshape::innerSetColor( double color )
{
	color_ = color;
}

double GLshape::getColor( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->color_;
}

void GLshape::setXOffset( const Conn* c, double xoffset )
{
	static_cast< GLshape * >( c->data() )->innerSetXOffset( xoffset );
}

void GLshape::innerSetXOffset( double xoffset )
{
	xoffset_ = xoffset;
}

double GLshape::getXOffset( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->xoffset_;
}

void GLshape::setYOffset( const Conn* c, double yoffset )
{
	static_cast< GLshape * >( c->data() )->innerSetYOffset( yoffset );
}

void GLshape::innerSetYOffset( double yoffset )
{
	yoffset_ = yoffset;
}

double GLshape::getYOffset( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->yoffset_;
}

void GLshape::setZOffset( const Conn* c, double zoffset )
{
	static_cast< GLshape * >( c->data() )->innerSetZOffset( zoffset );
}

void GLshape::innerSetZOffset( double zoffset )
{
	zoffset_ = zoffset;
}

double GLshape::getZOffset( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->zoffset_;
}

void GLshape::setLen( const Conn* c, double len )
{
	static_cast< GLshape * >( c->data() )->innerSetLen( len );
}

void GLshape::innerSetLen( double len )
{
	len_ = len;
}

double GLshape::getLen( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->len_;
}

void GLshape::setShapeType( const Conn* c, int shapetype )
{
	static_cast< GLshape * >( c->data() )->innerSetShapeType( shapetype );
}

void GLshape::innerSetShapeType( int shapetype )
{
	shapetype_ = shapetype;
}

int GLshape::getShapeType( Eref e )
{
	return static_cast< const GLshape* >( e.data() )->shapetype_;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void GLshape::reinitFunc( const Conn* c, ProcInfo info )
{
}

void GLshape::processFunc( const Conn* c, ProcInfo info )
{
}
