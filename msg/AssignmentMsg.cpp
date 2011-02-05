/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MsgManager.h"
#include "AssignmentMsg.h"

#include "ReduceBase.h"
#include "ReduceMax.h"
#include "../shell/Shell.h"

Id AssignmentMsg::id_;

AssignmentMsg::AssignmentMsg( Eref e1, Eref e2, MsgId mid )
	: Msg( e1.element(), e2.element(), mid, id_ ),
	i1_( e1.index() ), 
	i2_( e2.index() )
{
	;
}

AssignmentMsg::~AssignmentMsg()
{
	MsgManager::dropMsg( mid() );
}

void sendAckBack( const ProcInfo* p, MsgId mid, DataId i2 )
{
	static const Finfo* ackFinfo = 
		Shell::initCinfo()->findFinfo( "handleAck" );
	static const DestFinfo* df = 
		dynamic_cast< const DestFinfo* >( ackFinfo );
	static const FuncId ackFid = df->getFid();

	assert( df );
	Qinfo retq( ackFid, i2, 2 * sizeof( unsigned int ), 0 );
	MsgFuncBinding mfb( mid, ackFid );

	unsigned int ack[2];
	ack[0] = Shell::myNode();
	ack[1] = Shell::OkStatus;
	retq.addToQbackward( p, mfb, reinterpret_cast< char* >( ack ) );
}

void AssignmentMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		if ( e2_->dataHandler()->isDataHere( i2_ ) &&
			p->execThread( e2_->id(), i2_.data() ) )
		{
			const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
			PrepackedBuffer pb( arg + sizeof( Qinfo ) );
			f->op( Eref( e2_, i2_ ), q, pb.data() );
		}
		// Whether or not this node or thread handled it, we do need to
		// send back an ack from the node. So do it on thread 0.
		if ( p->threadIndexInGroup == 0 )
			sendAckBack( p, q->mid(), i2_ );
		return;
	} 

	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) &&
		p->execThread( e1_->id(), i1_.data() ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

Id AssignmentMsg::id() const
{
	return id_;
}

FullId AssignmentMsg::findOtherEnd( FullId f ) const
{
	if ( f.id() == e1() ) {
		return FullId( e2()->id(), i2_ );
	}
	if ( f.id() == e2() ) {
		return FullId( e1()->id(), i1_ );
	}
	return FullId::bad();
}

/// Dummy. We should never be copying assignment messages.
Msg* AssignmentMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	assert( 0 );
	return 0;
}

void AssignmentMsg::addToQ( const Element* src, Qinfo& q,
	const ProcInfo* p, MsgFuncBinding i, const char* arg ) const
{
	if ( e1_ == src && i1_ == q.srcIndex() ) {
		q.addToQforward( p, i, arg );
	} else if ( e2_ == src && i2_ == q.srcIndex() ) {
		q.addToQbackward( p, i, arg );
	}
}
