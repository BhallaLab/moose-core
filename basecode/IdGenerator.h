/**********************************************************************
** This program is part of 'MOOSE', the
** Multiscale Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ID_GENERATOR_H
#define _ID_GENERATOR_H

class IdGenerator
{
public:
	IdGenerator();

	IdGenerator( unsigned int id, unsigned int node );

	Id next();

	//~ void done();

	bool isGlobal() const;

private:
	unsigned int id_;

	bool global_;
};

#endif // _ID_GENERATOR_H
