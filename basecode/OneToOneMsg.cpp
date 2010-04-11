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

OneToOneMsg::OneToOneMsg( Element* e1, Element* e2 )
	: Msg( e1, e2 )
{
	;
}

void OneToOneMsg::exec( const char* arg, const ProcInfo* p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );
	if ( q->isForward() ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e2_, q->srcIndex() ), arg );
	} else {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, q->srcIndex() ), arg );
	}
}

