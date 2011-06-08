/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
/*
#include "MsgManager.h"
#include "SingleMsg.h"
#include "DiagonalMsg.h"
#include "OneToOneMsg.h"
#include "OneToAllMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"
#include "ReduceMsg.h"
#include "ReduceFinfo.h"
*/
#include "AssignmentMsg.h"
#include "AssignVecMsg.h"
#include "Shell.h"
// #include "Dinfo.h"

////////////////////////////////////////////////////////////////////////
// Functions for handling field set/get and func calls
////////////////////////////////////////////////////////////////////////

void Shell::handleSet( const Eref& e, const Qinfo* q, 
	Id id, DataId d, FuncId fid, PrepackedBuffer arg )
{
	if ( q->addToStructuralQ() )
		return;
	Eref er( id(), d );
	shelle_->clearBinding ( lowLevelSetGet()->getBindIndex() );
	Eref sheller( shelle_, 0 );
	Msg* m;
	
	if ( arg.isVector() ) {
		m = new AssignVecMsg( Msg::setMsg, sheller, er.element() );
	} else {
		m = new AssignmentMsg( Msg::setMsg, sheller, er );
		// innerSet( er, fid, arg.data(), arg.dataSize() );
	}
	shelle_->addMsgAndFunc( m->mid(), fid, lowLevelSetGet()->getBindIndex() );
	if ( myNode_ == 0 )
		lowLevelSetGet()->send( sheller, &p_, arg );
}

// Static function, used for developer-code triggered SetGet functions.
// Should only be issued from master node.
// This is a blocking function, and returns only when the job is done.
// mode = 0 is single value set, mode = 1 is vector set, mode = 2 is
// to set the entire target array to a single value.
void Shell::dispatchSet( const ObjId& tgt, FuncId fid, const char* args,
	unsigned int size )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	PrepackedBuffer buf( args, size );
	s->innerDispatchSet( sheller, tgt, fid, buf );
}

// regular function, does the actual dispatching.
void Shell::innerDispatchSet( Eref& sheller, const ObjId& tgt, 
	FuncId fid, const PrepackedBuffer& buf )
{
	initAck();
		requestSet()->send( sheller, &p_, tgt.id, tgt.dataId, fid, buf );
	waitForAck();
}

// Static function.
void Shell::dispatchSetVec( const ObjId& tgt, FuncId fid, 
	const PrepackedBuffer& pb )
{
	Eref sheller = Id().eref();
	Shell* s = reinterpret_cast< Shell* >( Id().eref().data() );
	s->innerDispatchSet( sheller, tgt, fid, pb );
}

const vector< char* >& Shell::dispatchGet( 
	const Eref& sheller, 
	const ObjId& tgt, FuncId fid,
	const PrepackedBuffer& buf )
{
	clearGetBuf();
	gettingVector_ = ( buf.numEntries() >= 1 );
	if ( gettingVector_ )
		getBuf_.resize( buf.numEntries() );
	else
		getBuf_.resize( 1 );
	numGetVecReturns_ = 0;
	initAck();
		requestSet()->send( sheller, &p_, tgt.id, tgt.dataId, fid, buf );
	waitForGetAck();

	return getBuf_;
}


/**
 * This operates on the worker node. It handles the Get request from
 * the master node, and dispatches if need to the local object.
 */
void Shell::handleGet( const Eref& e, const Qinfo* q, 
	Id id, DataId index, FuncId fid, unsigned int numTgt )
{
	if ( q->addToStructuralQ() )
		return;

	Eref sheller( shelle_, 0 );
	Eref tgt( id(), index );
	FuncId retFunc = receiveGet()->getFid();

	shelle_->clearBinding( lowLevelSetGet()->getBindIndex() );
	if ( numTgt > 1 ) {
		Msg* m = new AssignVecMsg( Msg::setMsg, sheller, tgt.element() );
		shelle_->addMsgAndFunc( m->mid(), fid, lowLevelSetGet()->getBindIndex() );
		if ( myNode_ == 0 ) {
			//Need to find numTgt
			vector< FuncId > rf( numTgt, retFunc );
			const char* temp = reinterpret_cast< const char* >( &rf[0] );
			PrepackedBuffer pb( temp, sizeof( FuncId ), numTgt );
			lowLevelSetGet()->send( sheller, &p_, pb );
		}
	} else {
		Msg* m = new AssignmentMsg( Msg::setMsg, sheller, tgt );
		shelle_->addMsgAndFunc( m->mid(), fid, lowLevelSetGet()->getBindIndex());
		if ( myNode_ == 0 ) {
			PrepackedBuffer pb( 
				reinterpret_cast< const char* >( &retFunc), 
				sizeof( FuncId ) );
			lowLevelSetGet()->send( sheller, &p_, pb );
		}
	}
}

void Shell::recvGet( const Eref& e, const Qinfo* q, PrepackedBuffer pb )
{
	if ( myNode_ == 0 ) {
		if ( gettingVector_ ) {
			assert( q->mid() != Msg::badMsg );
			Element* tgtElement = Msg::getMsg( q->mid() )->e2();
			Eref tgt( tgtElement, q->srcIndex() );
			assert ( tgt.linearIndex() < getBuf_.size() );
			char*& c = getBuf_[ tgt.linearIndex() ];
			c = new char[ pb.dataSize() ];
			memcpy( c, pb.data(), pb.dataSize() );
/*
if ( tgt.linearIndex() > 68570 && tgt.linearIndex() < 68640 ) {
	cout << "linearIndex = " << tgt.linearIndex() << 
		", DataId: " << q->srcIndex() << 
		", val = " << *reinterpret_cast< const double* >( c ) << endl;
}
*/
			// cout << myNode_ << ": Shell::recvGet[" << tgt.linearIndex() << "]= (" << pb.dataSize() << ", " <<  *reinterpret_cast< const double* >( c ) << ")\n";
		} else  {
			assert ( getBuf_.size() == 1 );
			char*& c = getBuf_[ 0 ];
			c = new char[ pb.dataSize() ];
			memcpy( c, pb.data(), pb.dataSize() );
		}
		++numGetVecReturns_;
	}
}

/*
void Shell::lowLevelRecvGet( PrepackedBuffer pb )
{
	cout << "Shell::lowLevelRecvGet: If this is being used, then fix\n";
	relayGet.send( Eref( shelle_, 0 ), &p_, myNode(), OkStatus, pb );
}
*/

////////////////////////////////////////////////////////////////////////

void Shell::clearGetBuf()
{
	for ( vector< char* >::iterator i = getBuf_.begin(); 
		i != getBuf_.end(); ++i )
	{
		if ( *i != 0 ) {
			delete[] *i;
			*i = 0;
		}
	}
	getBuf_.resize( 1, 0 );
}
