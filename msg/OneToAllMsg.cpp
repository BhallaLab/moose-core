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
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		if ( e2_->dataHandler()->numDimensions() == 1 ) {
			DataHandler::iterator end = e2_->dataHandler()->end();
			for ( DataHandler::iterator i = e2_->dataHandler()->begin();
				i != end; ++i )
				f->op( Eref( e2_, i ), arg );
		} else if ( e2_->dataHandler()->numDimensions() == 2 ) {
			// The first dimension is partitioned between nodes
			DataHandler::iterator end = e2_->dataHandler()->end();
			for ( DataHandler::iterator i = e2_->dataHandler()->begin();
				i != end; ++i ) {
				for ( unsigned int j = 0; j < e2_->dataHandler()->numData2( i ); ++j )
					f->op( Eref( e2_, DataId( i, j ) ), arg );
			}
		}
	} else {
		if ( e1_->dataHandler()->isDataHere( i1_ ) ) {
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
	if ( f.id() == e1() )
		return FullId( e2()->id(), 0 );
	else if ( f.id() == e2() )
		return FullId( e1()->id(), i1_ );
	
	return FullId( Id(), DataId::bad() );
}
