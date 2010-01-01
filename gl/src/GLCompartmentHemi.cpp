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

#include "GLCompartmentHemi.h"
#include "GLCompartmentHemiData.h"

GLCompartmentHemi::GLCompartmentHemi( osg::Vec3 centre, osg::Vec3f orientation, double radius, double incrementAngle )
	:
	centre_( centre ),
	radius_( radius ),
	incrementAngle_( incrementAngle ),
	orientation_( orientation )
{ 
	init();
}

GLCompartmentHemi::GLCompartmentHemi( const GLCompartmentHemiData& data, double incrementAngle )
	:
	centre_( data.centre[0], data.centre[1], data.centre[2] ),
	radius_( data.radius ),
	incrementAngle_( incrementAngle ),
	orientation_( data.orientation[0], data.orientation[1], data.orientation[2] )
{
	init();
}

GLCompartmentHemi::~GLCompartmentHemi()
{
	geometry_ = NULL;
	vertices_ = NULL;
	normals_ = NULL;
}

CompartmentType GLCompartmentHemi::getCompartmentType()
{
	return COMP_HEMISPHERE;
}

void GLCompartmentHemi::init()
{
	geometry_ = new osg::Geometry;
	vertices_ = new osg::Vec3Array;
	normals_ = new osg::Vec3Array;

	constructGeometry();
	
	geometry_->setVertexArray( vertices_ );
	geometry_->setNormalArray( normals_ );
	geometry_->setNormalBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
}

void GLCompartmentHemi::constructGeometry()
{
	std::vector< double > angles;
	for ( double i = 0; i <= 360 - incrementAngle_; i += incrementAngle_ )
		angles.push_back( 2*M_PI*i / 360 );
	angles.push_back( 0 );
	
	// prepare quaternion to rotate each point according to specified orientation
	osg::Vec3f initial( 0.0f, 0.0f, 1.0f );
	initial.normalize();

	orientation_.normalize();

	osg::Vec3f axis = initial ^ orientation_;
	axis.normalize();
	osg::Quat quatRotation( acos( initial * orientation_ ), axis );

	// add vertex at tip first
	vertices_->push_back( rotateTranslatePoint( osg::Vec3( 0, 0, radius_ ),
						    quatRotation,
						    centre_ ) );
	
	for ( unsigned int i = 0; i < angles.size()-1; ++i)
	{
		osg::DrawElementsUInt* faces = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);

		vertices_->push_back( rotateTranslatePoint( osg::Vec3( radius_ * cos( angles[i] ),
								       radius_ * sin( angles[i] ),
								       0 ),
							    quatRotation,
							    centre_ ) );
		vertices_->push_back( rotateTranslatePoint( osg::Vec3( radius_ * cos( angles[i+1] ),
								       radius_ * sin( angles[i+1] ),
								       0 ),
							    quatRotation,
							    centre_ ) );
		unsigned int j;
		for ( j = 1; j <= 9; ++j )
		{
	  
			faces = new osg::DrawElementsUInt( osg::PrimitiveSet::QUADS, 0 );

			vertices_->push_back( rotateTranslatePoint( osg::Vec3( radius_ * sin( acos(j*0.1) ) * cos( angles[i] ),
									       radius_ * sin( acos(j*0.1) ) * sin( angles[i] ),
									       ( (j*0.1) * radius_ ) ),
								    quatRotation,
								    centre_ ) );
			vertices_->push_back( rotateTranslatePoint( osg::Vec3( radius_ * sin( acos(j*0.1) ) * cos( angles[i+1] ),
									       radius_ * sin( acos(j*0.1) ) * sin( angles[i+1] ),
									       ( (j*0.1) * radius_ ) ),
								    quatRotation,
								    centre_ ) );

			faces->push_back( 1 + i*20 + 1 + 2*(j-1) ); // 20 == 2 + 9*2 ; vertices on middle ring + two vertices per face added
			faces->push_back( 1 + i*20 + 0 + 2*(j-1) );
			faces->push_back( 1 + i*20 + 2 + 2*(j-1) );
			faces->push_back( 1 + i*20 + 3 + 2*(j-1) );

			geometry_->addPrimitiveSet( faces );

			osg::Vec3 normal = makeNormal( ( *vertices_ )[1 + i*20 + 1 + 2*(j-1)],
						       ( *vertices_ )[1 + i*20 + 0 + 2*(j-1)],
						       ( *vertices_ )[1 + i*20 + 2 + 2*(j-1)] );
			normals_->push_back( osg::Vec3( normal[0] * -1,
						       normal[1] * -1,
						       normal[2] * -1 ) );
	
			faces = new osg::DrawElementsUInt( osg::PrimitiveSet::QUADS, 0);


		}

		j = 9;
	
		faces = new osg::DrawElementsUInt( osg::PrimitiveSet::TRIANGLES, 0 );
		faces->push_back( 1 + i*20 + 3 + 2*(j-1) );
		faces->push_back( 1 + i*20 + 2 + 2*(j-1) );
		faces->push_back( 0 );

		geometry_->addPrimitiveSet( faces );

		osg::Vec3 normal = makeNormal( ( *vertices_ )[1 + i*20 + 3 + 2*(j-1)],
					       ( *vertices_ )[1 + i*20 + 2 + 2*(j-1)],
					       ( *vertices_ )[0] );
		normals_->push_back( osg::Vec3( normal[0] * -1,
						normal[1] * -1,
						normal[2] * -1 ) );
	}
}

