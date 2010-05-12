/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DiagonalMsg.h"

DiagonalMsg::DiagonalMsg( Element* e1, Element* e2 )
	: Msg( e1, e2, id_ ), stride_( 1 )
{
	;
}

DiagonalMsg::~DiagonalMsg()
{
	MsgManager::dropMsg( mid() );
}

/**
 * This is sort of thread-safe, as there is only ever one target for
 * any given input. Furthermore, this target will always be unique:
 * no other input will hit the same target. So we can partition the
 * message exec operation among sources as we like.
 */
void DiagonalMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		int src = q->srcIndex().data();
		int dest = src + stride_;
		if ( dest >= 0 && e2_->dataHandler()->isDataHere( dest ) ) {
			const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e2_, dest ), arg );
		}
	} else {
		// Here we are stuck a bit. I will assume srcIndex is now for e2.
		int src = q->srcIndex().data();
		int dest = src - stride_;
		if ( dest >= 0 && e1_->dataHandler()->isDataHere( dest ) ) {
			const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
			f->op( Eref( e1_, dest ), arg );
		}
	}
}

Id DiagonalMsg::id() const
{
	return id_;
}

void DiagonalMsg::setStride( int stride )
{
	stride_ = stride;
}

int DiagonalMsg::getStride() const
{
	return stride_;
}
