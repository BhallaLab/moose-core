/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "ReduceBase.h"
#include "ReduceFinfo.h"
#include "ReduceMsg.h"

Id ReduceMsg::managerId_;

ReduceMsg::ReduceMsg( MsgId mid, Eref e1, Element* e2, const ReduceFinfoBase* rfb  )
	: Msg( mid, e1.element(), e2, ReduceMsg::managerId_ ),
		i1_( e1.index() ),
		rfb_( rfb )
{
	;
}

ReduceMsg::~ReduceMsg()
{
	;
}

void ReduceMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		/*
		FuncId fid = *reinterpret_cast< const FuncId* >( arg + sizeof( Qinfo ) );
		// Create Reduce operator.
		/// ReduceFinfoBase rfb_ is built into the Msg.
		ReduceBase* r = rfb_->makeReduce( e2_, q->fid() );
		or
		*/
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		ReduceBase* r = rfb_->makeReduce( ObjId( e1_->id(), i1_ ), f );
		// Qinfo::addToReduceQ( Eref( e1_, i1_ ), rfb_, r, p->threadIndexInGroup() );
		Qinfo::addToReduceQ( r, p->threadIndexInGroup );
		DataHandler* d2 = e2_->dataHandler();
		for ( DataHandler::iterator i = d2->begin(); i != d2->end(); ++i )
		{
			if ( p->execThread( e2_->id(),i.index().data() ) ) {
				// This fills up the first pass of reduce operations.
				r->primaryReduce( ObjId( e2_->id(), i.index() ) );
			}
		}
	}
	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) &&
		p->execThread( e1_->id(), i1_.data() ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

/*
// when parsing the ReduceQ:
First go through all the ReduceBase ptrs for a given slot.
Then do the Allgather or Gather depending on whether the elm is a
	Global or a local. Also depends on elm field. hm.
Then use the rfb and elm info to assign the value using
	digestReduce.

*/

Id ReduceMsg::managerId() const
{
	return ReduceMsg::managerId_;
}

ObjId ReduceMsg::findOtherEnd( ObjId f ) const
{
	if ( f.id() == e1() ) {
		return ObjId( e2()->id(), 0 );
	}
	if ( f.id() == e2() ) {
		return ObjId( e1()->id(), i1_ );
	}
	return ObjId::bad();
}

/// Dummy. We should never be copying assignment messages.
Msg* ReduceMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	assert( 0 );
	return 0;
}

void ReduceMsg::addToQ( const Element* src, Qinfo& q,
	const ProcInfo* p, MsgFuncBinding i, const char* arg ) const
{
	if ( e1_ == src && i1_ == q.srcIndex() ) {
		q.addToQforward( p, i, arg );
	} else if ( e2_ == src ) { // Not sure if we should allow back_msg here.
		q.addToQbackward( p, i, arg ); 
	}
}

///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* ReduceMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< ReduceMsg, DataId > i1(
		"i1",
		"DataId of source Element.",
		&ReduceMsg::getI1
	);

	static Finfo* msgFinfos[] = {
		&i1,		// readonly value
	};

	static Cinfo msgCinfo (
		"ReduceMsg",	// name
		Msg::initCinfo(),				// base class
		msgFinfos,
		sizeof( msgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< short >()
	);

	return &msgCinfo;
}

static const Cinfo* reduceMsgCinfo = ReduceMsg::initCinfo();

/**
 * Return the first DataId
 */
DataId ReduceMsg::getI1() const
{
	return i1_;
}
