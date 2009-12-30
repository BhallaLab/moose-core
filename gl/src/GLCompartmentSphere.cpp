/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <osg/Vec3d>
#include <osg/ref_ptr>
#include <osg/Geometry>

#include <vector>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846             
#endif

#include "GLCompartmentSphere.h"
#include "GLCompartmentSphereData.h"

GLCompartmentSphere::GLCompartmentSphere( osg::Vec3 centre, double radius, double incrementAngle )
	:
	centre_( centre ),
	radius_( radius ),
	incrementAngle_( incrementAngle )
{ 
	init();
}

GLCompartmentSphere::GLCompartmentSphere( const GLCompartmentSphereData& data, double incrementAngle )
	:
	centre_( data.centre[0], data.centre[1], data.centre[2] ),
	radius_( data.radius ),
	incrementAngle_( incrementAngle )
{
	init();
}

void GLCompartmentSphere::init()
{
	geometry_ = new osg::Geometry;
	vertices_ = new osg::Vec3Array;
	normals_ = new osg::Vec3Array;

	addHemisphericalCap( true );
	addHemisphericalCap( false ); // two hemi-spheres make a sphere
	
	geometry_->setVertexArray( vertices_ );
	geometry_->setNormalArray( normals_ );
	geometry_->setNormalBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
}

GLCompartmentSphere::~GLCompartmentSphere()
{
	geometry_ = NULL;
	vertices_ = NULL;
	normals_ = NULL;
}

CompartmentType GLCompartmentSphere::getCompartmentType()
{
	return COMP_SPHERE;
}

void GLCompartmentSphere::addHemisphericalCap( bool leftEndP )
{
	int oldSizeVertices = vertices_->size();

	std::vector< double > angles;
	for ( double i = 0; i <= 360 - incrementAngle_; i += incrementAngle_ )
		angles.push_back( 2*M_PI*i / 360 );
	angles.push_back( 0 );

	double neighbourSign;
	if ( leftEndP )
	{
		neighbourSign = -1;
	}
	else
	{
		neighbourSign = 1;
	}

	// add vertex at tip first
	vertices_->push_back( osg::Vec3( centre_[0],
					 centre_[1],
					 centre_[2] + neighbourSign * radius_ ) );
	
	for ( unsigned int i = 0; i < angles.size()-1; ++i)
	{
		osg::DrawElementsUInt* faces = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);

		vertices_->push_back( osg::Vec3( centre_[0] + radius_ * cos( angles[i] ),
						 centre_[1] + radius_ * sin( angles[i] ),
						 centre_[2] ) );
		vertices_->push_back( osg::Vec3( centre_[0] + radius_ * cos( angles[i+1] ),
						 centre_[1] + radius_ * sin( angles[i+1] ),
						 centre_[2] ) );

		unsigned int j;
		for ( j = 1; j <= 9; ++j )
		{
			faces = new osg::DrawElementsUInt( osg::PrimitiveSet::QUADS, 0 );

			vertices_->push_back( osg::Vec3( centre_[0] + radius_ * sin( acos(j*0.1) ) * cos( angles[i] ),
							 centre_[1] + radius_ * sin( acos(j*0.1) ) * sin( angles[i] ),
							 centre_[2] + neighbourSign * ( (j*0.1) * radius_ ) ) );
			vertices_->push_back( osg::Vec3( centre_[0] + radius_ * sin( acos(j*0.1) ) * cos( angles[i+1] ),
							 centre_[1] + radius_ * sin( acos(j*0.1) ) * sin( angles[i+1] ),
							 centre_[2] + neighbourSign * ( (j*0.1) * radius_ ) ) );

			faces->push_back( oldSizeVertices + 1 + i*20 + 1 + 2*(j-1) ); // 20 == 2 + 9*2 ; vertices on middle ring + two vertices per face added
			faces->push_back( oldSizeVertices + 1 + i*20 + 0 + 2*(j-1) );
			faces->push_back( oldSizeVertices + 1 + i*20 + 2 + 2*(j-1) );
			faces->push_back( oldSizeVertices + 1 + i*20 + 3 + 2*(j-1) );

			geometry_->addPrimitiveSet( faces );

			osg::Vec3 normal = makeNormal( ( *vertices_ )[oldSizeVertices + 1 + i*20 + 1 + 2*(j-1)],
						       ( *vertices_ )[oldSizeVertices + 1 + i*20 + 0 + 2*(j-1)],
						       ( *vertices_ )[oldSizeVertices + 1 + i*20 + 2 + 2*(j-1)] );
			normals_->push_back(osg::Vec3( normal[0] * -1 * neighbourSign,
						       normal[1] * -1 * neighbourSign,
						       normal[2] * -1 * neighbourSign ) );
	
			faces = new osg::DrawElementsUInt( osg::PrimitiveSet::QUADS, 0);
		}

		j = 9;
	
		faces = new osg::DrawElementsUInt( osg::PrimitiveSet::TRIANGLES, 0 );
		faces->push_back( oldSizeVertices + 1 + i*20 + 3 + 2*(j-1) );
		faces->push_back( oldSizeVertices + 1 + i*20 + 2 + 2*(j-1) );
		faces->push_back( oldSizeVertices );

		geometry_->addPrimitiveSet( faces );

		osg::Vec3 normal = makeNormal( ( *vertices_ )[oldSizeVertices + 1 + i*20 + 3 + 2*(j-1)],
					       ( *vertices_ )[oldSizeVertices + 1 + i*20 + 2 + 2*(j-1)],
					       ( *vertices_ )[oldSizeVertices] );
		normals_->push_back( osg::Vec3( normal[0] * -1 * neighbourSign,
						normal[1] * -1 * neighbourSign,
						normal[2] * -1 * neighbourSign ) );
	}
}

