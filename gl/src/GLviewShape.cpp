/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Shape>

#include "GLviewShape.h"

GLviewShape::GLviewShape( unsigned int id, std::string strPathName, 
			  double x, double y, double z,
			  double len, int shapetype )
	:
	xoffset_( 0.0 ),
	yoffset_( 0.0 ),
	zoffset_( 0.0 )
{
	id_ = id;
	strPathName_ = strPathName;
	x_ = x; 
	y_ = y;
	z_ = z;
	len_ = len;
	shapetype_ = shapetype;
	
	if ( shapetype_ == CUBE )
	{
		box_ = new osg::Box( osg::Vec3( x_ + len_/2,
						y_ + len_/2,
						z_ + len_/2 ),
				     len_ );
		drawable_ = new osg::ShapeDrawable( box_ );
	}
	else
	{
		sphere_ = new osg::Sphere( osg::Vec3( x_ + len_/2,
						      y_ + len_/2,
						      z_ + len_/2 ),
					   len_/2 );
		drawable_ = new osg::ShapeDrawable( sphere_ );
	}

	geode_ = new osg::Geode();	
	geode_->addDrawable( drawable_ );
}

osg::ref_ptr< osg::Geode > GLviewShape::getGeode()
{
	return geode_;
}

void GLviewShape::setColor( osg::Vec4 color )
{
	drawable_->setColor( color );
}

void GLviewShape::move( double xoffset, double yoffset, double zoffset )
{
	xoffset_ = xoffset;
	yoffset_ = yoffset;
	zoffset_ = zoffset;

	if ( shapetype_ == CUBE )
	{
		box_->setCenter( osg::Vec3( x_ + len_/2 + xoffset_,
					    y_ + len_/2 + yoffset_,
					    z_ + len_/2 + zoffset_ ) );
	}
	else
	{
		sphere_->setCenter( osg::Vec3( x_ + len_/2 + xoffset_,
					       y_ + len_/2 + yoffset_,
					       z_ + len_/2 + zoffset_ ) );
	}
}

void GLviewShape::resize( double len )
{
	len_ = len;

	if ( shapetype_ == CUBE )
		box_->setHalfLengths( osg::Vec3( len_/2, len_/2, len_/2 ) );
	else
		sphere_->setRadius( len_/2 );
}

