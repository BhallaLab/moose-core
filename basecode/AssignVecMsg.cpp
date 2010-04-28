/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "AssignVecMsg.h"

AssignVecMsg::AssignVecMsg( Eref e1, Element* e2, MsgId mid )
	: Msg( e1.element(), e2, mid ),
	i1_( e1.index() )
{
	;
}

void AssignVecMsg::exec( const char* arg, const ProcInfo *p ) const
{
	const Qinfo *q = ( reinterpret_cast < const Qinfo * >( arg ) );

	if ( q->isForward() ) {
		PrepackedBuffer pb( arg + sizeof( Qinfo ) );
		if ( pb.dataSize() == 0 )
			Qinfo::reportQ();
		// cout << Shell::myNode() << ": AssignVecMsg::exec: pb.size = " << pb.size() << ", dataSize = " << pb.dataSize() << ", numEntries = " << pb.numEntries() << endl;
		DataHandler* d2 = e2_->dataHandler();
		const OpFunc* f = e2_->cinfo()->getOpFunc( q->fid() );
		if ( d2->numDimensions() == 1 ) {
			for ( DataHandler::iterator i = d2->begin(); 
				i != d2->end(); ++i ) {

				// This is nasty. We assume that none of the op funcs
				// will actually use the Qinfo. Better to set up a
				// temporary with the original Qinfo at the start, and
				// copy the pb[i] into the correct place there.
				f->op( Eref( e2_, i ), pb[i] - sizeof( Qinfo ) );
			}
			return;
		}
		if ( d2->numDimensions() == 2 ) {
			unsigned int k = 0;
			for ( unsigned int i = 0; 
				i < e2_->dataHandler()->numData1(); ++i )
				for ( unsigned int j = 0; 
					j < e2_->dataHandler()->numData2( i ); ++j )
				// This is nasty. We assume that none of the op funcs
				// will actually use the Qinfo. 
					f->op( Eref( e2_, DataId( i, j ) ), 
						pb[ k++ ] - sizeof( Qinfo ) );
			// Problem here. 2nd dimension size may vary. Do we do a 
			// single stripe, or the entire array? Assume latter.
		}
	}
	if ( !q->isForward() && e1_->dataHandler()->isDataHere( i1_ ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( q->fid() );
		f->op( Eref( e1_, i1_ ), arg );
	}
}
