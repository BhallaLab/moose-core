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
#include "GLCompartmentDiskData.h"

class GLCompartmentDisk : public GLCompartment
{
public:
	GLCompartmentDisk( osg::Vec3 centre, osg::Vec3f orientation, double radius, double incrementAngle );
	GLCompartmentDisk( const GLCompartmentDiskData& data, double incrementAngle );
	~GLCompartmentDisk();
	
	CompartmentType getCompartmentType();

private:
	void init();
	void constructGeometry();

	osg::Vec3 centre_;
	osg::Quat quatRotation_;
	double radius_;
	double incrementAngle_;
};

