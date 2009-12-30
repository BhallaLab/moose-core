/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLCOMPHEMIDATA_H
#define GLCOMPHEMIDATA_H

struct GLCompartmentHemiData
{
	float centre[3];
	float orientation[3];
	double radius;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version)
	{
		ar & centre[0];
		ar & centre[1];
		ar & centre[2];

		ar & orientation[0];
		ar & orientation[1];
		ar & orientation[2];

		ar & radius;
	}
};

#endif // GLCOMPHEMIDATA_H
