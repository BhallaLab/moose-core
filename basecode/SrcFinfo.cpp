/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

/**
 * This set of classes define Message Sources. Their main job is to supply 
 * a type-safe send operation, and to provide typechecking for it.
 */

SrcFinfo::SrcFinfo( const string& name, const string& doc, ConnId c )
	: Finfo( name, doc ), c_( c ), funcIndex_( 0 )
{ ; }

void SrcFinfo::registerOpFuncs(
		map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
{
	;
}

unsigned int SrcFinfo::registerSrcFuncIndex( unsigned int current )
{
	funcIndex_ = current;
	return current + 1;
}

unsigned int SrcFinfo::registerConn( unsigned int current )
{
	if ( c_ >= current )
		return c_ + 1;
	return current;
}

/////////////////////////////////////////////////////////////////////
/**
 * SrcFinfo0 sets up calls without any arguments.
 */
SrcFinfo0::SrcFinfo0( const string& name, const string& doc, ConnId c )
	: SrcFinfo( name, doc, c )
{ ; }

void SrcFinfo0::send( Eref e ) const {
	e.asend( getConnId(), getFuncIndex(), 0, 0 );
}

void SrcFinfo0::sendTo( Eref e, Id target ) const
{
	unsigned int temp = target.index();
	/*
	char temp[ sizeof( unsigned int ) ];
	*reinterpret_cast< unsigned int* >( temp ) = target.index();
	*/
	e.tsend( getConnId(), getFuncIndex(), target, 
	reinterpret_cast< const char* >( &temp ), 0 );
}
