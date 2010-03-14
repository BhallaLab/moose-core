/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"

/**
 * This set of classes define Message Sources. Their main job is to supply 
 * a type-safe send operation, and to provide typechecking for it.
 */

SharedFinfo::SharedFinfo( const string& name, const string& doc,
	const Finfo** entries, unsigned int numEntries )
	: Finfo( name, doc )
{ 
	for ( unsigned int i = 0; i < numEntries; ++i )
	{
		const SrcFinfo* s = dynamic_cast< const SrcFinfo* >( entries[i] );
		if ( s != 0 )
			src_.push_back( s );
		else
			dest_.push_back( entries[i] );
	}
}

void SharedFinfo::registerOpFuncs(
		map< string, FuncId >& fnames, vector< OpFunc* >& funcs ) 
{
	;
}

BindIndex SharedFinfo::registerBindIndex( BindIndex current )
{
	return current;
}

bool SharedFinfo::checkTarget( const Finfo* target ) const
{
	const SharedFinfo* tgt = dynamic_cast< const SharedFinfo* >( target );
	if ( tgt ) {
		if ( src_.size() != tgt->dest_.size() && 
			dest_.size() != tgt->src_.size() )
			return 0;
		for ( unsigned int i = 0; i < src_.size(); ++i ) {
			if ( !src_[i]->checkTarget( tgt->dest_[i] ) )
				return 0;
		}
		for ( unsigned int i = 0; i < tgt->src_.size(); ++i ) {
			if ( !tgt->src_[i]->checkTarget( dest_[i] ) )
				return 0;
		}

		return 1;
	}
	return 0;
}

bool SharedFinfo::addMsg( const Finfo* target, MsgId mid,
	Id src, Id dest ) const
{
	if ( !checkTarget( target ) )
		return 0;
	const SharedFinfo* tgt = dynamic_cast< const SharedFinfo* >( target );
	for ( unsigned int i = 0; i < src_.size(); ++i ) {
		if ( !src_[i]->addMsg( tgt->dest_[i], mid, src, dest ) ) {
			// Should never happen. The checkTarget should preclude this.
			cerr << "Error:SharedFinfo::addMsg: Failed between " <<
				src << " and " << dest << ", unrecoverable\n";
			exit(0);
		}
	}
	for ( unsigned int i = 0; i < tgt->src_.size(); ++i ) {
		if ( !tgt->src_[i]->addMsg( dest_[i], mid, src, dest ) ) {
			// Should never happen. The checkTarget should preclude this.
			cerr << "Error:SharedFinfo::addMsg: Failed between " <<
				src << " and " << dest << ", unrecoverable\n";
			exit( 0 );
		}
	}
	return 0;
}

/////////////////////////////////////////////////////////////////////
