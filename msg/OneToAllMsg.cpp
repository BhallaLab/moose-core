/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "OneToAllMsg.h"

Id OneToAllMsg::managerId_;

OneToAllMsg::OneToAllMsg( MsgId mid, Eref e1, Element* e2 )
	: 
		Msg( mid, e1.element(), e2, OneToAllMsg::managerId_ ),
		i1_( e1.index() )
{
	;
}

OneToAllMsg::~OneToAllMsg()
{
	/*
	cout << "Deleting OneToAllMsg from " << e1_->getName() << " to " <<
		e2_->getName() << endl;
		*/
	;
}

/**
 * Need to revisit to handle nodes
 */
void OneToAllMsg::exec( const Qinfo* q, const double* arg, FuncId fid ) 
	const
{
	if ( q->src().element() == e1_ ) {
		if ( q->src().dataId == i1_ ) {
			const OpFunc* f = e2_->cinfo()->getOpFunc( fid );
			e2_->dataHandler()->foreach( f, e2_, q, arg, 0 );
			/*
			DataHandler::iterator end = e2_->dataHandler()->end();
			for ( DataHandler::iterator i = e2_->dataHandler()->begin();
				i != end; ++i ) {
				if ( q->execThread( e2_->id(), i.index().data() ) )
				{
					f->op( Eref( e2_, i.index() ), q, arg );
					// cout << ":" << q->threadNum() << ": ObjId = " << ObjId( e2_->id(), i.index() ) << endl;
				}
			}
			*/
		}
	} else {
		if ( e1_->dataHandler()->isDataHere( i1_ )  &&
			q->execThread( e1_->id(), i1_.value() ) )
		{
			const OpFunc* f = e1_->cinfo()->getOpFunc( fid );
			f->op( Eref( e1_, i1_ ), q, arg );
		}
	}
}

Eref OneToAllMsg::firstTgt( const Eref& src ) const 
{
	if ( src.element() == e1_ )
		return Eref( e2_, 0 );
	else if ( src.element() == e2_ )
		return Eref( e1_, i1_ );
	return Eref( 0, 0 );
}

Id OneToAllMsg::managerId() const
{
	return OneToAllMsg::managerId_;
}

ObjId OneToAllMsg::findOtherEnd( ObjId f ) const
{
	if ( f.id() == e1() ) {
		if ( f.dataId == i1_ )
			return ObjId( e2()->id(), 0 );
		else
			return ObjId( e2()->id(), DataId::bad );
	} else if ( f.id() == e2() ) {
		return ObjId( e1()->id(), i1_ );
	}
	
	return ObjId::bad;
}

Msg* OneToAllMsg::copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const
{
	const Element* orig = origSrc();
	if ( n <= 1 ) {
		OneToAllMsg* ret = 0;
		if ( orig == e1() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), Eref( newSrc(), i1_ ), newTgt() );
			ret->e1()->addMsgAndFunc( ret->mid(), fid, b );
		} else if ( orig == e2() ) {
			ret = new OneToAllMsg( Msg::nextMsgId(), Eref( newTgt(), i1_ ), newSrc() );
			ret->e2()->addMsgAndFunc( ret->mid(), fid, b );
		} else {
			assert( 0 );
		}
		return ret;
	} else {
		// Here we need a SliceMsg which goes from one 2-d array to another.
		cout << "Error: OneToAllMsg::copy: SliceToSliceMsg not yet implemented\n";
		return 0;
	}
}

unsigned int OneToAllMsg::srcToDestPairs(
	vector< DataId >& src, vector< DataId >& dest ) const
{
	/*
	class GetDataId: public OpFuncDummy
	{
		public:
			GetDataId( vector< DataId >* vec )
				: vec_( vec )
			{;}
			void op( const Eref& e, const Qinfo* q, const double* buf) const
			{
				vec_->push_back( e.index() );
			}
		private:
			vector< DataId >* vec_;
	};
	*/

	dest.resize( 0 );
	DataIdExtractor di( &dest );

	e2_->dataHandler()->foreach( &di, 0, 0, 0, 0 );
	src.resize( dest.size(), i1_ );

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
	*/

	return dest.size();
}

///////////////////////////////////////////////////////////////////////
// Here we set up the MsgManager portion of the class.
///////////////////////////////////////////////////////////////////////

const Cinfo* OneToAllMsg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< OneToAllMsg, DataId > i1(
		"i1",
		"DataId of source Element.",
		&OneToAllMsg::getI1
	);

	static Finfo* msgFinfos[] = {
		&i1,		// readonly value
	};

	static Cinfo msgCinfo (
		"OneToAllMsg",	// name
		Msg::initCinfo(),				// base class
		msgFinfos,
		sizeof( msgFinfos ) / sizeof( Finfo* ),	// num Fields
		new Dinfo< short >()
	);

	return &msgCinfo;
}

static const Cinfo* assignmentMsgCinfo = OneToAllMsg::initCinfo();

/**
 * Return the first DataId
 */
DataId OneToAllMsg::getI1() const
{
	return i1_;
}
