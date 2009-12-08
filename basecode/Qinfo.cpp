/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

// Declaration of static field
vector< vector< char > > Qinfo::q_;

Qinfo::Qinfo( FuncId f, DataId srcIndex, 
	unsigned int size, bool useSendTo, bool isForward )
	:	m_( 0 ), 
		useSendTo_( useSendTo ), 
		isForward_( isForward ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}

Qinfo::Qinfo( FuncId f, DataId srcIndex, unsigned int size )
	:	m_( 0 ), 
		useSendTo_( 0 ), 
		isForward_( 1 ), 
		f_( f ), 
		srcIndex_( srcIndex ),
		size_( size )
{;}


/*
void Qinfo::addToQ( vector< char >& q, const char* arg ) const
{
	unsigned int origSize = q.size();
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}
*/

void Qinfo::expandSize()
{
	size_ += sizeof( DataId );
}

// Static func
void Qinfo::setNumQs( unsigned int n, unsigned int reserve )
{
	q_.resize( n );
	for ( unsigned int i = 0; i < n; ++i )
		q_[i].reserve( reserve );
}

// static func
// Note that it does not advance the buffer.
void Qinfo::hackForSendTo( const Qinfo* q, const char* buf )
{
	const DataId* tgtIndex = 
		reinterpret_cast< const DataId* >( buf + sizeof( Qinfo ) +
		q->size() - sizeof( DataId ) );

	Element* tgt;
	if ( q->isForward() )
		tgt = Msg::getMsg( q->m_ )->e2();
	else 
		tgt = Msg::getMsg( q->m_ )->e1();
	const OpFunc* func = tgt->cinfo()->getOpFunc( q->fid() );
	func->op( Eref( tgt, *tgtIndex ), buf );
}

// Static func
void Qinfo::clearQ( Qid qId )
{
	assert( qId < q_.size() );
	vector< char >& q = q_[qId];
	const char* buf = &q[0];
	while ( buf && buf < &q.back() )
	{
		const Qinfo *q = reinterpret_cast< const Qinfo* >( buf );
		if ( q->useSendTo() ) {
			hackForSendTo( q, buf );
		} else {
			const Msg* m = Msg::getMsg( q->m_ );
			m->exec( buf );
		}
		buf += sizeof( Qinfo ) + q->size();
	}
	q_.resize( 0 );
}

// Non-static: copies itself onto queue.
// qid specifies which queue to use.
void Qinfo::addToQ( Qid qId, MsgId mid, bool isForward, const char* arg )
	const
{
	assert( qId < q_.size() );

	vector< char >& q = q_[qId];
	unsigned int origSize = q.size();
	q.resize( origSize + sizeof( Qinfo ) + size_ );
	char* pos = &( q[origSize] );
	memcpy( pos, this, sizeof( Qinfo ) );
	( reinterpret_cast< Qinfo* >( pos ) )->setForward( isForward );
	memcpy( pos + sizeof( Qinfo ), arg, size_ );
}
