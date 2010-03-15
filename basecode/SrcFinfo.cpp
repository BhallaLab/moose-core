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

SrcFinfo::SrcFinfo( const string& name, const string& doc, BindIndex b )
	: Finfo( name, doc ), bindIndex_( b )
{ ; }

void SrcFinfo::registerFinfo( Cinfo* c )
{
	bindIndex_ = c->registerBindIndex();
}

/*
void SrcFinfo::registerOpFuncs(
		map< string, FuncId >& fnames, vector< OpFunc* >& funcs )
{
	;
}

BindIndex SrcFinfo::registerBindIndex( BindIndex current )
{
	bindIndex_ = current;
	return current + 1;
}
*/

bool SrcFinfo::checkTarget( const Finfo* target ) const
{
	const DestFinfo* d = dynamic_cast< const DestFinfo* >( target );
	if ( d ) {
		return d->getOpFunc()->checkFinfo( this );
	}
	return 0;
}

bool SrcFinfo::addMsg( const Finfo* target, MsgId mid, Element* src ) const
{
	const DestFinfo* d = dynamic_cast< const DestFinfo* >( target );
	if ( d ) {
		if ( d->getOpFunc()->checkFinfo( this ) ) {
			src->addMsgAndFunc( mid, d->getFid(), bindIndex_ );
			return 1;
		}
	}
	return 0;
}

/////////////////////////////////////////////////////////////////////
/**
 * SrcFinfo0 sets up calls without any arguments.
 */
SrcFinfo0::SrcFinfo0( const string& name, const string& doc, BindIndex b )
	: SrcFinfo( name, doc, b )
{ ; }

void SrcFinfo0::send( Eref e, const ProcInfo* p, bool isForward ) const {
	// First arg is useSendTo, second arg is isForward, last arg is size
	Qinfo q( 0, isForward, e.index(), 0 );
	e.element()->asend( q, getBindIndex(), p, 0 ); // last arg is data
}

void SrcFinfo0::sendTo( Eref e, const ProcInfo* p, 
	const FullId& target, bool isForward ) const
{
	Qinfo q( 1, isForward, e.index(), 0 );
	e.element()->tsend( q, getBindIndex(), p, 0, target );
}
