/**********************************************************************
** This program is part of 'MOOSE', the
** Multiscale Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "Id.h"
#include "IdGenerator.h"
#include "moose.h"
#include "shell/Shell.h"

IdGenerator::IdGenerator( )
	:
	id_( Id::badId().id() ),
	global_( false )
{ ; }

IdGenerator::IdGenerator( unsigned int id, unsigned int node )
	:
	id_( id ),
	global_( node == Id::GlobalNode )
{ ; }

Id IdGenerator::next()
{
	if ( global_ ) {
		Id id( id_, 0 );
		if ( Shell::myNode() == 0 )
			id = Id::newId();
		id_++;
		id.setGlobal();
		return id;
	} else {
		return Id::newId();
	}
}

bool IdGenerator::isGlobal() const
{
	return global_;
}
