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
#include "GLCompartmentRectData.h"

class GLCompartmentRect : public GLCompartment
{
public:
	GLCompartmentRect( osg::Vec3 corner1, osg::Vec3 corner2, osg::Vec3 corner3, osg::Vec3 corner4 );
	GLCompartmentRect( const GLCompartmentRectData& data );
	~GLCompartmentRect();
	
	CompartmentType getCompartmentType();

private:
	void init();
	void constructGeometry();

	osg::Vec3 corner1_;
	osg::Vec3 corner2_;
	osg::Vec3 corner3_;
	osg::Vec3 corner4_;
};

