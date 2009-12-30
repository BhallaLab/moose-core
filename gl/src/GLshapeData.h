/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef GLSHAPEDATA_H
#define GLSHAPEDATA_H

struct GLshapeData
{
	double color;
	double xoffset;
	double yoffset;
	double zoffset;
	double len; // diameter if spherical

	template< typename Archive >
	void serialize( Archive& ar, const unsigned int version )
	{
		ar & color;
		ar & xoffset;
		ar & yoffset;
		ar & zoffset;
		ar & len;
	}
};

#endif // GLSHAPEDATA_H


