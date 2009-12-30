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
#include "GLCompartmentSphereData.h"

class GLCompartmentSphere : public GLCompartment
{
public:
	GLCompartmentSphere( osg::Vec3 centre, double radius, double incrementAngle );
	GLCompartmentSphere( const GLCompartmentSphereData& data, double incrementAngle );
	~GLCompartmentSphere();

	CompartmentType getCompartmentType();

private:
	void init();
	void addHemisphericalCap( bool leftEndP );
  
	osg::Vec3 centre_;
	double radius_;
	double incrementAngle_;
};

