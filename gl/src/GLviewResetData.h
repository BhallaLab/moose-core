/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLVIEWRESETDATA_H
#define GLVIEWRESETDATA_H

struct GLviewShapeResetData
{
	unsigned int id;
	std::string strPathName;
	double x, y, z;
	int shapetype;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version )
	{
		ar & id;
		ar & strPathName;
		ar & x;
		ar & y;
		ar & z;
		ar & shapetype;
	}	     
};

struct GLviewResetData
{
	double bgcolorRed;
	double bgcolorGreen;
	double bgcolorBlue;
	std::string strPathName;

	// this will be the maximum size (absolute value) of
	// our elements along any dimension
	double maxsize;
	
	std::vector< GLviewShapeResetData > vecShapes;

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version )
	{
		ar & bgcolorRed;
		ar & bgcolorGreen;
		ar & bgcolorBlue;
		ar & strPathName;
		ar & maxsize;
		ar & vecShapes;
	}
};

#endif // GLVIEWRESETDATA_H
