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

const BindIndex SrcFinfo::BadBindIndex = 65535;

SrcFinfo::SrcFinfo( const string& name, const string& doc )
	: Finfo( name, doc ), bindIndex_( BadBindIndex )
{ ; }

void SrcFinfo::registerFinfo( Cinfo* c )
{
	bindIndex_ = c->registerBindIndex();
}


BindIndex SrcFinfo::getBindIndex() const 
{
	// Treat this assertion as a warning that the SrcFinfo is being used
	// without initialization.
	assert( bindIndex_ != BadBindIndex );
	return bindIndex_;
}

void SrcFinfo::setBindIndex( BindIndex b )
{
	bindIndex_ = b;
}

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
SrcFinfo0::SrcFinfo0( const string& name, const string& doc )
	: SrcFinfo( name, doc )
{ ; }

void SrcFinfo0::send( const Eref& e, ThreadId threadNum ) const {
	Qinfo::addToQ( e.objId(), getBindIndex(), threadNum, 0, 0 );
	/*
	Qinfo q( e.index(), 0, 0 );
	e.element()->asend( q, getBindIndex(), p, 0 ); // last arg is data
	*/
}

/*
void SrcFinfo0::sendTo( const Eref& e, const ProcInfo* p, 
	const ObjId& target ) const
{
	// Qinfo( eindex, size, useSendTo );
	Qinfo q( e.index(), 0, 1 );
	e.element()->tsend( q, getBindIndex(), p, 0, target );
}
*/

void SrcFinfo0::fastSend( const Eref& e, ThreadId threadNum ) const
{
	Qinfo qi( e.objId(), getBindIndex(), threadNum, 0, 0 );
	e.element()->exec( &qi, 0 );
}
