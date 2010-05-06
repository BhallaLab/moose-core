/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "AssignmentMsg.h"

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

void AssignmentMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() && e2_->dataHandler()->isDataHere( i2_ ) ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e2_, i2_ ), arg );
		return;
	} 

	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}

Id AssignmentMsg::id() const
{
	return id_;
}
