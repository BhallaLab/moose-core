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
#include "GLCompartmentHemiData.h"

class GLCompartmentHemi : public GLCompartment
{
public:
	GLCompartmentHemi( osg::Vec3 centre, osg::Vec3f orientation, double radius, double incrementAngle );
	GLCompartmentHemi( const GLCompartmentHemiData& data, double incrementAngle );
	~GLCompartmentHemi();

	CompartmentType getCompartmentType();

private:
	void init();
	void constructGeometry();
	
	osg::Vec3 centre_;
	double radius_;
	double incrementAngle_;
	osg::Vec3f orientation_;
};

