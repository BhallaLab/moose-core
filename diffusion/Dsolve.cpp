/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SparseMatrix.h"
#include "KinSparseMatrix.h"
#include "ZombiePoolInterface.h"
#include "DiffPoolVec.h"
#include "Dsolve.h"

Dsolve::Dsolve()
{;}

Dsolve::~Dsolve()
{;}

unsigned int Dsolve::getNumVarPools() const
{
	return 0;
}

void Dsolve::setPath( const Eref& e, string v )
{;}

string Dsolve::getPath( const Eref& e ) const
{
	return "foo";
}

void zombifyModel( const Eref& e, const vector< Id >& elist )
{
	;
}


void unZombifyModel( const Eref& e )
{
		;
}
