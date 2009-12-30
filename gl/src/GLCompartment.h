/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLCOMPARTMENT_H
#define GLCOMPARTMENT_H

#include "GLCompartmentType.h"

class GLCompartment
{
 public:
	osg::ref_ptr< osg::Geometry > getGeometry();
	void setColor( osg::Vec4 color );
	
	virtual CompartmentType getCompartmentType() = 0;
	
	virtual ~GLCompartment() {}

 protected:
	osg::Vec3 rotateTranslatePoint( osg::Vec3 position, osg::Quat& quatRotation, osg::Vec3& translation );
	double distance( const osg::Vec3& P1, const osg::Vec3& P2 );
	double distance( const float& x1, const float& y1, const float& z1,
			 const float& x2, const float& y2, const float& z2 );
	double distance( const double& x1, const double& y1, const double& z1,
			 const double& x2, const double& y2, const double& z2 );
	osg::Vec3 makeNormal( const osg::Vec3& P1, const osg::Vec3& P2, const osg::Vec3& P3 );

	osg::ref_ptr< osg::Geometry > geometry_;
	osg::ref_ptr< osg::Vec3Array > vertices_;
	osg::ref_ptr< osg::Vec3Array > normals_;
};

#endif // GLCOMPARTMENT_H
