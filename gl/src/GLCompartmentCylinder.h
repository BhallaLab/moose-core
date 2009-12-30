/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "GLCompartment.h"
#include "GLCompartmentCylinderData.h"

class GLCompartmentCylinder : public GLCompartment
{
public:
	GLCompartmentCylinder( osg::Vec3 endPoint1, osg::Vec3 endPoint2, double radius, double incrementAngle );
	GLCompartmentCylinder( const GLCompartmentCylinderData& data, double incrementAngle );
	~GLCompartmentCylinder();

	void addHalfJointToNeighbour( GLCompartmentCylinder* neighbour );
	void closeOpenEnds();
	
	CompartmentType getCompartmentType();

	bool isPointInsideCylinder( osg::Vec3& testPoint );

	osg::ref_ptr< osg::Vec3Array > ringRight;
	osg::ref_ptr< osg::Vec3Array > ringLeft;

private:
	void init();
	void constructGeometry();

	osg::Vec3 position_;
	osg::Quat quatRotation_;
	double length_;
	double radius_;
	double incrementAngle_;
	bool isLeftEndClosed_;
	bool isRightEndClosed_;

	void addHemisphericalCap( bool leftEndP );
};

