/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Finfo::Finfo( const string& name, const string& doc )
	: name_( name ), doc_( doc )
{
	;
}

const string& Finfo::name( ) const
{
	return name_;
}

//////////////////////////////////////////////////////////////////////////
DestFinfo::~DestFinfo() {
	delete func_;
}

DestFinfo::DestFinfo( const string& name, const string& doc, 
	OpFunc* func )
	: Finfo( name, doc ), func_( func )
{
	;
}

void DestFinfo::registerOpFuncs( 
	map< string, FuncId >& fnames, vector< OpFunc* >& funcs
	)
{
	map< string, FuncId >::iterator i = fnames.find( name() );
	if ( i != fnames.end() ) {
		funcs[ i->second ] = func_;
	} else {
		unsigned int size = funcs.size();
		fnames[ name() ] = size;
		funcs.push_back( func_ );
	}
}

BindIndex DestFinfo::registerBindIndex( BindIndex current )
{
	return current;
}
