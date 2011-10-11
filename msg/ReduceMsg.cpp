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
#include "../shell/Shell.h"

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

void ReduceMsg::exec( const Qinfo* q, const double* arg, FuncId fid ) const
{
	if ( q->src().element() == e1_ ) {
		const OpFunc* f = e2_->cinfo()->getOpFunc( fid );
		ReduceBase* r = rfb_->makeReduce( ObjId( e1_->id(), i1_ ), f );
		Qinfo::addToReduceQ( r, q->threadNum() );
		// DataHandler* d2 = e2_->dataHandler();
		//unsigned int count = 0;
		vector< DataId > vec;
		DataIdExtractor di( &vec );
		e2_->dataHandler()->foreach( &di, e2_, q, 0, 0, 0 );
		for ( vector< DataId >::const_iterator i = vec.begin(); 
			i != vec.end(); ++i ) {
			r->primaryReduce( ObjId( e2_->id(), *i ) );
			r->setInited();
		}
		/*
		for ( DataHandler::iterator i = d2->begin(); i != d2->end(); ++i )
		{
			if ( q->execThread( e2_->id(),i.index().data() ) ) {
				// This fills up the first pass of reduce operations.
				r->primaryReduce( ObjId( e2_->id(), i.index() ) );
				r->setInited();
				//++count;
			}
		}
		*/
		// cout << Shell::myNode() << ":" << q->threadNum() << " ReduceMsg::exec numPrimaryReduce = " << count << endl;
		// ReduceStats* rs = dynamic_cast< ReduceStats* >( r );
		// if ( rs ) {
		// cout << Shell::myNode() << ":" << q->threadNum() << " ReduceMsg::exec sum = " << rs->sum() << ", count = " << rs->count() << endl;
		// }
	} else if ( e1_->dataHandler()->isDataHere( i1_ ) &&
		q->execThread( e1_->id(), i1_.value() ) ) {
		const OpFunc* f = e1_->cinfo()->getOpFunc( fid );
		f->op( Eref( e1_, i1_ ), q, arg );
	}
}

Eref ReduceMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ )
		return Eref( e2_, 0 );
	else if ( src.element() == e2_ )
		return Eref( e1_, i1_ );
	return Eref( 0, 0 );
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
	return ObjId::bad;
}

/// Dummy. We should never be copying assignment messages.
Msg* ReduceMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	assert( 0 );
	return 0;
}

unsigned int ReduceMsg::srcToDestPairs(
	vector< DataId >& src, vector< DataId >& dest ) const
{
	dest.resize( 0 );
	DataIdExtractor di( &dest );

	Qinfo q;
	e2_->dataHandler()->foreach( &di, 0, &q, 0, 0, 0 );
	src.resize( dest.size(), i1_ );
	return dest.size();

	/*
	 unsigned int destRange = e2_->dataHandler()->totalEntries();
	src.resize( destRange, i1_ );
	dest.resize( destRange );
	unsigned int fd = e2_->dataHandler()->getFieldDimension();
	if ( fd <= 1 ) {
		for ( unsigned int i = 0; i < destRange; ++i )
			dest[i] = DataId( i );
	} else {
		for ( unsigned int i = 0; i < destRange; ++i )
			dest[i] = DataId( i / fd, i % fd );
	}

	return destRange;
	*/
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
