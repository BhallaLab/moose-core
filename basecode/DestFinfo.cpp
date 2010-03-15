/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

DestFinfo::~DestFinfo() {
	delete func_;
}

DestFinfo::DestFinfo( const string& name, const string& doc, 
	OpFunc* func )
	: Finfo( name, doc ), func_( func )
{
	;
}

/*
Finfo* DestFinfo::clone() const
{
	return new Destfinfo( *this );
}
*/

void DestFinfo::registerFinfo( Cinfo* c )
{
	fid_ = c->registerOpFunc( func_ );
}

/*
void DestFinfo::registerOpFuncs( 
	map< string, FuncId >& fnames, vector< OpFunc* >& funcs
	)
{
	map< string, FuncId >::iterator i = fnames.find( name() );
	if ( i != fnames.end() ) {
		funcs[ i->second ] = func_;
		fid_ = i->second;
	} else {
		unsigned int size = funcs.size();
		fnames[ name() ] = size;
		fid_ = size;
		funcs.push_back( func_ );
	}
}

BindIndex DestFinfo::registerBindIndex( BindIndex current )
{
	return current;
}
*/

const OpFunc* DestFinfo::getOpFunc() const
{
	return func_;
}

FuncId DestFinfo::getFid() const
{
	return fid_;
}
