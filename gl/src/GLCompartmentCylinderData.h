/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLCOMPCYLDATA_H
#define GLCOMPCYLDATA_H

struct GLCompartmentCylinderData
{
	double endPoint1[3];
	double endPoint2[3];
	double radius;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version)
	{
		ar & endPoint1[0];
		ar & endPoint1[1];
		ar & endPoint1[2];

		ar & endPoint2[0];
		ar & endPoint2[1];
		ar & endPoint2[2];

		ar & radius;
	}
};

#endif // GLCOMPCYLDATA_H
