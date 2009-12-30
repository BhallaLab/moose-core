/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLCOMPTRIDATA_H
#define GLCOMPTRIDATA_H

struct GLCompartmentTriData
{
	float corner1[3];
	float corner2[3];
	float corner3[3];

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version)
	{
		for( unsigned int i = 0; i < 3; ++i )
		{
			ar & corner1[i];
			ar & corner2[i];
			ar & corner3[i];
		}
	}
};

#endif // GLCOMPTRIDATA_H
