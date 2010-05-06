/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "Message.h"

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
	// arg += sizeof( Qinfo );
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		if ( e2_->dataHandler()->numDimensions() == 1 ) {
			DataHandler::iterator end = e2_->dataHandler()->end();
			for ( DataHandler::iterator i = e2_->dataHandler()->begin();
				i != end; ++i )
				f->op( Eref( e2_, i ), arg );

			/*
			for ( unsigned int i = 0; i < e2_->dataHandler()->numData(); ++i )
				f->op( Eref( e2_, i ), arg );
				*/
		} else if ( e2_->dataHandler()->numDimensions() == 2 ) {
			// The first dimension is partitioned between nodes
			DataHandler::iterator end = e2_->dataHandler()->end();
			for ( DataHandler::iterator i = e2_->dataHandler()->begin();
				i != end; ++i ) {
			// for ( unsigned int i = 0; i < e2_->dataHandler()->numData1(); ++i ) {
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

bool OneToAllMsg::add( Eref e1, const string& srcField, 
			Element* e2, const string& destField )
{
	FuncId funcId;
	const SrcFinfo* srcFinfo = validateMsg( e1.element(), srcField,
		e2, destField, funcId );

	if ( srcFinfo ) {
		Msg* m = new OneToAllMsg( e1, e2 );
		e1.element()->addMsgAndFunc( m->mid(), funcId, srcFinfo->getBindIndex() );
		return 1;
	}
	return 0;
}


Id OneToAllMsg::id() const
{
	return id_;
}
