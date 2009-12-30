#include <osg/Vec3>
#include <osg/ref_ptr>
#include <osg/Geometry>

#include "GLCompartment.h"

osg::ref_ptr< osg::Geometry > GLCompartment::getGeometry()
{
	return geometry_;
}

void GLCompartment::setColor( osg::Vec4 color )
{
	osg::Vec4Array* colors_ = new osg::Vec4Array;
	colors_->push_back( color );

	geometry_->setColorArray( colors_ );
	geometry_->setColorBinding( osg::Geometry::BIND_OVERALL );
}

osg::Vec3 GLCompartment::rotateTranslatePoint( osg::Vec3 position, osg::Quat& quatRotation, osg::Vec3& translation )
{
	position = quatRotation * position;

	position[0] += translation[0];
	position[1] += translation[1];
	position[2] += translation[2];

	return position;
}

osg::Vec3 GLCompartment::makeNormal( const osg::Vec3& P1, const osg::Vec3& P2, const osg::Vec3& P3 )
{
	osg::Vec3 U = osg::Vec3( P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2] );
	osg::Vec3 V = osg::Vec3( P3[0]-P1[0], P3[1]-P1[1], P3[2]-P1[2] );

	osg::Vec3 Normal;

	Normal[0] = U[1]*V[2] - U[2]*V[1];
	Normal[1] = U[2]*V[0] - U[0]*V[2];
	Normal[2] = U[0]*V[1] - U[1]*V[0];

	double mag = sqrt( Normal[0]*Normal[0] + Normal[1]*Normal[1] + Normal[2]*Normal[2] );

	Normal[0] /= mag;
	Normal[1] /= mag;
	Normal[2] /= mag;

	return Normal;
}

double GLCompartment::distance( const osg::Vec3& P1, const osg::Vec3& P2 )
{
	return sqrt( pow( P1[0] - P2[0], 2 ) +
		     pow( P1[1] - P2[1], 2 ) + 
		     pow( P1[2] - P2[2], 2 ) );
}

double GLCompartment::distance( const float& x1, const float& y1, const float& z1,
				const float& x2, const float& y2, const float& z2 )
{
	return sqrt( pow( x1 - x2, 2 ) +
		     pow( y1 - y2, 2 ) + 
		     pow( z1 - z2, 2 ) );
}

double GLCompartment::distance( const double& x1, const double& y1, const double& z1,
				const double& x2, const double& y2, const double& z2 )
{
	return sqrt( pow( x1 - x2, 2 ) +
		     pow( y1 - y2, 2 ) + 
		     pow( z1 - z2, 2 ) );
}
