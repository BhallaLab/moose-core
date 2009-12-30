/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef SMOLDYNSHAPEDATA_H
#define SMOLDYNSHAPEDATA_H

#include "GLCompartmentType.h"
#include "GLCompartmentCylinderData.h"
#include "GLCompartmentDiskData.h"
#include "GLCompartmentHemiData.h"
#include "GLCompartmentRectData.h"
#include "GLCompartmentSphereData.h"
#include "GLCompartmentTriData.h"

#include <boost/variant.hpp>
#include <boost/serialization/variant.hpp>

// An instance of this struct stores one shape (or Smoldyn
// 'surface'). The geometrical specification of the shape is stored in data.
// Supported shapes are hemispheres, spheres, disks, cylinders, triangles
// and rectangles.

struct SmoldynShapeData
{
	double color[4];
	std::string name;
	boost::variant< GLCompartmentCylinderData,
			GLCompartmentDiskData,
			GLCompartmentHemiData,
			GLCompartmentRectData,
			GLCompartmentSphereData,
			GLCompartmentTriData > data;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version)
	{
		ar & color[0];
		ar & color[1];
		ar & color[2];
		ar & color[3];
		ar & name;
		ar & data;
	}
};

#endif // SMOLDYNSHAPEDATA_H
