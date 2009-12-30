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
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846             
#endif

#include "GLCompartmentDisk.h"
#include "GLCompartmentDiskData.h"

GLCompartmentDisk::GLCompartmentDisk( osg::Vec3 centre, osg::Vec3f orientation,
				      double radius, double incrementAngle )
	:
	centre_( centre ),
	radius_( radius ),
	incrementAngle_( incrementAngle )
{
	osg::Vec3f initial( 0.0f, 0.0f, 1.0f );
	initial.normalize();
	
	orientation.normalize();

	osg::Quat::value_type angle = acos( initial * orientation );
	osg::Vec3f axis = initial ^ orientation;
	axis.normalize();

	quatRotation_ = osg::Quat( angle, axis );

	init();
}

GLCompartmentDisk::GLCompartmentDisk( const GLCompartmentDiskData& data, double incrementAngle )
	:
	centre_( data.centre[0], data.centre[1], data.centre[2] ),
	radius_( data.radius ),
	incrementAngle_( incrementAngle )	
{
	osg::Vec3f initial( 0.0f, 0.0f, 1.0f );
	initial.normalize();
	
	osg::Vec3f orientation( data.orientation[0], data.orientation[1], data.orientation[2] );
	orientation.normalize();

	osg::Quat::value_type angle = acos( initial * orientation );
	osg::Vec3f axis = initial ^ orientation;
	axis.normalize();

	quatRotation_ = osg::Quat( angle, axis );

	init();
}

GLCompartmentDisk::~GLCompartmentDisk()
{
	geometry_ = NULL;
	vertices_ = NULL;
	normals_ = NULL;
}

void GLCompartmentDisk::init()
{
	geometry_ = new osg::Geometry;
	vertices_ = new osg::Vec3Array;
	normals_ = new osg::Vec3Array;

	constructGeometry();

	geometry_->setVertexArray( vertices_ );
	geometry_->setNormalArray( normals_ );
	geometry_->setNormalBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
}

void GLCompartmentDisk::constructGeometry()
{
	std::vector< double > angles;
	for ( double i = 0; i <= 360 - incrementAngle_; i += incrementAngle_ )
		angles.push_back( 2*M_PI*i / 360 );
	angles.push_back( 0.0 );

	// push the centrepoint first
	vertices_->push_back( rotateTranslatePoint( osg::Vec3( 0.0, 0.0, 0.0 ),
						    quatRotation_,
						    centre_ ) );
    
	for ( unsigned int i = 0; i < angles.size(); ++i )
	{
		vertices_->push_back( rotateTranslatePoint( osg::Vec3( radius_*cos(angles[i]),
								       radius_*sin(angles[i]),
								       0.0 ),
							       quatRotation_,
							       centre_ ) );
	}

	for ( unsigned int j = 0; j < angles.size() - 1; ++j )
	{
		osg::DrawElementsUInt* faces = new osg::DrawElementsUInt( osg::PrimitiveSet::TRIANGLES, 0 );

		faces->push_back( 0 );
		faces->push_back( j + 1 );
		faces->push_back( j + 2 );

		geometry_->addPrimitiveSet( faces );

		normals_->push_back(makeNormal( ( *vertices_ )[0],
						( *vertices_ )[j+1],
						( *vertices_ )[j+2] ));
	}	
}

CompartmentType GLCompartmentDisk::getCompartmentType()
{
	return COMP_DISK;
}
