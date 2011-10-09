/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OneToOneMsg.h"

Id OneToOneMsg::managerId_;

OneToOneMsg::OneToOneMsg( MsgId mid, Element* e1, Element* e2 )
	: Msg( mid, e1, e2, OneToOneMsg::managerId_ )
{
	;
}

OneToOneMsg::~OneToOneMsg()
{
	;
}

void OneToOneMsg::exec( const Qinfo* q, const double* arg, FuncId fid) const
{
	unsigned int src = q->src().dataId.value(); // will also be dest index.
	if ( q->src().element() == e1_ ) {
		if ( e2_->dataHandler()->isDataHere( src ) &&
			q->execThread( e2_->id(), src ) )
		{
			const OpFunc* f = e2_->cinfo()->getOpFunc( fid );
			f->op( Eref( e2_, q->src().dataId ), q, arg );
		}
	} else {
		if ( e1_->dataHandler()->isDataHere( src ) &&
			q->execThread( e1_->id(), src ) )
		{
			const OpFunc* f = e1_->cinfo()->getOpFunc( fid );
			f->op( Eref( e1_, q->src().dataId ), q, arg );
		}
	}
}

/**
 * This is a little tricky because we might be mapping between
 * data entries and field entries here.
 * May wish also to apply to exec operations.
 * At this point, the effect of trying to go between regular
 * data entries and field entries is undefined.
 */
Eref OneToOneMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ ) {
		return Eref( e2_, src.index() );
		/*
		if ( e2_->dataHandler()->getFieldArraySize( 0 ) == 0 ) {
			if ( e2_->dataHandler()->isDataHere( src.index() ) )
				return Eref( e2_, src.index() );
		} else {
			if ( e2_->dataHandler()->getFieldArraySize( 0 ) > 
				src.index().data() )
			return Eref( e2_, DataId( 0, src.index().data() ) );
		}
		*/
	} else if ( src.element() == e2_ ) {
		return Eref( e1_, src.index() );
		/*
		if ( e2_->dataHandler()->getFieldArraySize( 0 ) == 0 ) {
			if ( e1_->dataHandler()->isDataHere( src.index() ) )
				return Eref( e1_, src.index() );
		} else {
			if ( e1_->dataHandler()->getFieldArraySize( 0 ) > 
				src.index().data() )
				return Eref( e1_, DataId( 0, src.index().data() ) );
		}
		*/
	}
	return Eref( 0, 0 );
}

Id OneToOneMsg::managerId() const
{
	return OneToOneMsg::managerId_;
}

ObjId OneToOneMsg::findOtherEnd( ObjId f ) const
{
	if ( f.id() == e1() )
		return ObjId( e2()->id(), f.dataId );
	else if ( f.id() == e2() )
		return ObjId( e1()->id(), f.dataId );
	
	return ObjId::bad;
}

unsigned int OneToOneMsg::srcToDestPairs(
	vector< DataId >& src, vector< DataId >& dest ) const
{
	Qinfo q;
	DataIdExtractor srcdi( &src );
	e1_->dataHandler()->foreach( &srcdi, 0, &q, 0, 0, 0 );
	DataIdExtractor destdi( &dest );
	e2_->dataHandler()->foreach( &destdi, 0, &q, 0, 0, 0 );
	assert ( src.size() == dest.size() );

	/*
	unsigned int srcRange = e1_->dataHandler()->totalEntries();
	unsigned int destRange = e2_->dataHandler()->totalEntries();
	unsigned int range = ( srcRange < destRange ) ? srcRange : destRange;

	src.resize( range );
	dest.resize( range );
	unsigned int fd1 = e2_->dataHandler()->getFieldDimension();
	unsigned int fd2 = e2_->dataHandler()->getFieldDimension();
	if ( fd1 != fd2 ) {
		cout << "Warning: OneToOneMsg::srcToDestPairs: Mismatch in fieldDimension of e1 and e2:\n";
		cout << e1_->getName() << " -> " << e2_->getName() << endl;
	}
	if ( fd1 == 0 ) fd1 = 1;
	if ( fd2 == 0 ) fd2 = 1;
	for ( unsigned int i = 0; i < range; ++i ) {
		src[i] = DataId( i / fd1, i % fd1 );
		dest[i] = DataId( i / fd2, i % fd2 );
	}
	*/

	return src.size();
}

Msg* OneToOneMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc();
	// This works both for 1-copy and for n-copies
	OneToOneMsg* ret = 0;
	if ( orig == e1() ) {
		ret = new OneToOneMsg( Msg::nextMsgId(), newSrc(), newTgt() );
		ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
	} else if ( orig == e2() ) {
		ret = new OneToOneMsg( Msg::nextMsgId(), newTgt(), newSrc() );
		ret->e2()->addMsgAndFunc( ret->mid(), fid, b );
	} else
		assert( 0 );
	// ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
	return ret;
}

///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* OneToOneMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions. Nothing here.
	///////////////////////////////////////////////////////////////////

	static Cinfo msgCinfo (
		"OneToOneMsg",	// name
		Msg::initCinfo(),				// base class
		0,								// Finfo array
		0,								// Num Fields
		new Dinfo< short >()
	);

	return &msgCinfo;
}

static const Cinfo* oneToOneMsgCinfo = OneToOneMsg::initCinfo();

