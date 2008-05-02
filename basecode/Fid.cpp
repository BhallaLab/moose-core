/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../shell/Shell.h"

//////////////////////////////////////////////////////////////
//	Fid creation
//////////////////////////////////////////////////////////////

Fid::Fid()
	: id_(), fieldNum_( 0 )
{;}

Fid::Fid( Id id, const string& fname )
	: id_( id ), fieldNum_( 0 )
{
	assert( id.good() );
	const Finfo* f = id()->findFinfo( fname );
	assert( f != 0 );
	fieldNum_ = f->msg();
}

Fid::Fid( const string& name )
{
	string::size_type pos = name.find_last_of( "/" );
	id_ = Id( name.substr( 0, pos ) );
	assert( id_.good() );

	const Finfo* f = id_()->findFinfo( name.substr( pos + 1 ) );
	assert( f != 0 );
	fieldNum_ = f->msg();
}

Fid::Fid( Id id, int f )
	: id_( id ), fieldNum_( f )
{
	assert( id.good() );
}

const Finfo* Fid::finfo()
{
	return id_()->findFinfo( fieldNum_ );
}

string Fid::fieldName() const 
{
	assert( id_.good() );
	const Finfo* f = id_()->findFinfo( fieldNum_ );
	assert( f );
	return f->name();
}

string Fid::name() const
{
	return Shell::eid2path( id_ ) + "/" + fieldName();
}
