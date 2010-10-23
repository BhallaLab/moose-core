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
#include "OneToAllMsg.h"

Id OneToAllMsg::id_;


OneToAllMsg::OneToAllMsg( Eref e1, Element* e2 )
	: 
		Msg( e1.element(), e2, id_ ),
		i1_( e1.index() )
{
	;
}

OneToAllMsg::~OneToAllMsg()
{
	MsgManager::dropMsg( mid() );
}

/**
 * Need to revisit to handle nodes
 */
void OneToAllMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	unsigned int threadIndex = ( mid() + p->threadIndexInGroup ) % p->numThreadsInGroup;
	if ( q->isForward() ) {
		DataHandler::iterator end = e2_->dataHandler()->end();
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		for ( DataHandler::iterator i = e2_->dataHandler()->begin();
			i != end; ++i ) {
			if ( ( ++threadIndex == p->numThreadsInGroup ) ) {
			// Partition portions of target among threads.
					f->op( Eref( e2_, i.index() ), arg );
					threadIndex = 0;
			}
		}
	} else {
		// More or less randomly, one thread will deal with this.
		if ( threadIndex == 0 && e1_->dataHandler()->isDataHere( i1_ ) ) {
			const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e1_, i1_ ), arg );
		}
	}
}

bool OneToAllMsg::isMsgHere( const Qinfo& q ) const
{
	if ( q.isForward() )
		return ( i1_ == q.srcIndex() );
	return 1; // Going the other way, any of the indices can send the msg.
}

Id OneToAllMsg::id() const
{
	return id_;
}

FullId OneToAllMsg::findOtherEnd( FullId f ) const
{
	if ( f.id() == e1() ) {
		if ( f.dataId == i1_ )
			return FullId( e2()->id(), 0 );
		else
			return FullId( e2()->id(), DataId::bad() );
	} else if ( f.id() == e2() ) {
		return FullId( e1()->id(), i1_ );
	}
	
	return FullId::bad();
}

Msg* OneToAllMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc();
	if ( n <= 1 ) {
		OneToAllMsg* ret;
		if ( orig == e1() )
			ret = new OneToAllMsg( Eref( newSrc(), i1_ ), newTgt() );
		else if ( orig == e2() )
			ret = new OneToAllMsg( Eref( newTgt(), i1_ ), newSrc() );
		else
			assert( 0 );
		ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
		return ret;
	} else {
		// Here we need a SliceMsg which goes from one 2-d array to another.
		cout << "Error: OneToAllMsg::copy: SliceToSliceMsg not yet implemented\n";
		return 0;
	}
}
