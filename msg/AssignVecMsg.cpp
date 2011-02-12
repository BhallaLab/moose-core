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
#include "AssignVecMsg.h"

// Defined in AssignmentMsg.cpp
extern void sendAckBack( const ProcInfo* p, MsgId mid, DataId i2 );

Id AssignVecMsg::id_;

AssignVecMsg::AssignVecMsg( MsgId mid, Eref e1, Element* e2 )
	: Msg( mid, e1.element(), e2, id_ ),
	i1_( e1.index() )
{
	;
}

AssignVecMsg::~AssignVecMsg()
{
	MsgManager::dropMsg( mid() );
}

void AssignVecMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		PrepackedBuffer pb( arg + sizeof( Qinfo ) );
		// cout << Shell::myNode() << ": AssignVecMsg::exec: pb.size = " << pb.size() << ", dataSize = " << pb.dataSize() << ", numEntries = " << pb.numEntries() << endl;
		DataHandler* d2 = e2_->dataHandler();
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		for ( DataHandler::iterator i = d2->begin(); i != d2->end(); ++i )
		{
			if ( p->execThread( e2_->id(),i.index().data() ) ) {
				// Note that j might not go in sequential order, as it
				// depends on locally allocated parts of the vector.
				// But we assume that pb[j] has the entire data block and
				// so we need to pick selected entries from it.
				// Note also that this is independent of the # of dimensions
				// or whether the DataHandler is a FieldDataHandler.
				f->op( Eref( e2_, i.index() ), q, pb[ i.linearIndex() ] );
			}
		}
		if ( p->threadIndexInGroup == 0 )
			sendAckBack( p, q->mid(), 0 );
	}
	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) &&
		p->execThread( e1_->id(), i1_.data() ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

Id AssignVecMsg::id() const
{
	return id_;
}

FullId AssignVecMsg::findOtherEnd( FullId f ) const
{
	if ( f.id() == e1() ) {
		return FullId( e2()->id(), 0 );
	}
	if ( f.id() == e2() ) {
		return FullId( e1()->id(), i1_ );
	}
	return FullId::bad();
}

/// Dummy. We should never be copying assignment messages.
Msg* AssignVecMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	assert( 0 );
	return 0;
}

void AssignVecMsg::addToQ( const Element* src, Qinfo& q,
	const ProcInfo* p, MsgFuncBinding i, const char* arg ) const
{
	if ( e1_ == src && i1_ == q.srcIndex() ) {
		q.addToQforward( p, i, arg );
	} else if ( e2_ == src ) { // Not sure if we should allow back_msg here.
		q.addToQbackward( p, i, arg ); 
	}
}
