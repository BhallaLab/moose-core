/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <osg/Vec3>
#include <osg/ref_ptr>
#include <osg/Geometry>

#include <vector>

#include "GLCompartmentRect.h"
#include "GLCompartmentRectData.h"

GLCompartmentRect::GLCompartmentRect( osg::Vec3 corner1, osg::Vec3 corner2,
				      osg::Vec3 corner3, osg::Vec3 corner4 )
	:
	corner1_( corner1 ),
	corner2_( corner2 ),
	corner3_( corner3 ),
	corner4_( corner4 )
{
	init();
}

GLCompartmentRect::GLCompartmentRect( const GLCompartmentRectData& data )
	:
	corner1_( data.corner1[0], data.corner1[1], data.corner1[2] ),
	corner2_( data.corner2[0], data.corner2[1], data.corner2[2] ),
	corner3_( data.corner3[0], data.corner3[1], data.corner3[2] ),
	corner4_( data.corner4[0], data.corner4[1], data.corner4[2] )
{
	init();
}

GLCompartmentRect::~GLCompartmentRect()
{
	geometry_ = NULL;
	vertices_ = NULL;
	normals_ = NULL;
}

void GLCompartmentRect::init()
{
	geometry_ = new osg::Geometry;
	vertices_ = new osg::Vec3Array;
	normals_ = new osg::Vec3Array;

	constructGeometry();

	geometry_->setVertexArray( vertices_ );
	geometry_->setNormalArray( normals_ );
	geometry_->setNormalBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
}

void GLCompartmentRect::constructGeometry()
{
	vertices_->push_back( corner1_ );
	vertices_->push_back( corner2_ );
	vertices_->push_back( corner3_ );
	vertices_->push_back( corner4_ );

	osg::DrawElementsUInt* faces = new osg::DrawElementsUInt( osg::PrimitiveSet::QUADS, 0 );

	faces->push_back( 0 );
	faces->push_back( 1 );
	faces->push_back( 2 );
	faces->push_back( 3 );

	geometry_->addPrimitiveSet( faces );

	normals_->push_back(makeNormal( ( *vertices_ )[0],
					( *vertices_ )[1],
					( *vertices_ )[2] ));
}

CompartmentType GLCompartmentRect::getCompartmentType()
{
	return COMP_RECT;
}
