/**********************************************************************
 ** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SingleMsg.h"
#include "DiagonalMsg.h"
#include "OneToOneMsg.h"
#include "OneToAllMsg.h"
#include "SparseMatrix.h"
#include "SparseMsg.h"

#include "../shell/Shell.h"

///////////////////////////////////////////////////////////////////////////

// Static field declaration.
vector< Msg* > Msg::msg_;
vector< MsgId > Msg::garbageMsg_;
const MsgId Msg::bad = 0;
const MsgId Msg::setMsg = 1;
Id Msg::msgManagerId_;
vector< Id > msgMgrs;

Msg::Msg( MsgId mid, Element* e1, Element* e2, Id mgrId )
	: mid_( mid), e1_( e1 ), e2_( e2 )
{
	// assert( mid_ < msg_.size() );
	if ( mid_ >= msg_.size() ) {
		msg_.resize( mid + 1, 0 );
	}
	msg_[mid_] = this;
	e1->addMsg( mid_ );
	e2->addMsg( mid_ );
	/*
	MsgDataHandler * mdh = dynamic_cast< MsgDataHandler* >( 
		mgrId.element()->dataHandler() );
	assert( mdh );
	mdh->addMid( mid_ );
	*/
}

Msg::~Msg()
{
	msg_[ mid_ ] = 0;
	e1_->dropMsg( mid_ );
	e2_->dropMsg( mid_ );

	if ( mid_ > 1 )
		garbageMsg_.push_back( mid_ );
}

/**
 * Returns a MsgId for assigning to a new Msg.
 */
MsgId Msg::nextMsgId()
{
	MsgId ret;
	ret = msg_.size();
	msg_.push_back( 0 );
	return ret;
}

void Msg::deleteMsg( MsgId mid )
{
	assert( mid < msg_.size() );
	Msg* m = msg_[ mid ];
	if ( m != 0 )
		delete m;
	msg_[ mid ] = 0;
}

/**
 * Initialize the Null location in the Msg vector.
 */
void Msg::initNull()
{
	assert( msg_.size() == 0 );
	nextMsgId(); // Set aside entry 0 for badMsg;
	nextMsgId(); // Set aside entry 1 for setMsg;
}

const Msg* Msg::getMsg( MsgId m )
{
	assert( m < msg_.size() );
	return msg_[ m ];
}

Msg* Msg::safeGetMsg( MsgId m )
{
	if ( m == bad )
		return 0;
	if ( m < msg_.size() )
		return msg_[ m ];
	return 0;
}

Eref Msg::manager() const
{
	return Eref( ( this->managerId() )(), DataId( mid_ ) );
}

// Static func
unsigned int Msg::numMsgs()
{
	return msg_.size();
}

/**
 * Return the first element id
 */
Id Msg::getE1() const
{
	return e1_->id();
}

/**
 * Return the second element id
 */
Id Msg::getE2() const
{
	return e2_->id();
}

vector< string > Msg::getSrcFieldsOnE1() const
{
	vector< pair< BindIndex, FuncId > > ids;
	vector< string > ret;

	e1_->getFieldsOfOutgoingMsg( mid_, ids );

	for ( unsigned int i = 0; i < ids.size(); ++i ) {
		string name = e1_->cinfo()->srcFinfoName( ids[i].first );
		if ( name == "" ) {
			cout << "Error: Msg::getSrcFieldsOnE1: Failed to find field on msg " <<
			e1_->getName() << "-->" << e2_->getName() << endl;
		} else {
			ret.push_back( name );
		}
	}
	return ret;
}

vector< string > Msg::getDestFieldsOnE2() const
{
	vector< pair< BindIndex, FuncId > > ids;
	vector< string > ret;

	e1_->getFieldsOfOutgoingMsg( mid_, ids );

	for ( unsigned int i = 0; i < ids.size(); ++i ) {
		string name = e2_->cinfo()->destFinfoName( ids[i].second );
		if ( name == "" ) {
			cout << "Error: Msg::getDestFieldsOnE2: Failed to find field on msg " <<
			e1_->getName() << "-->" << e2_->getName() << endl;
		} else {
			ret.push_back( name );
		}
	}
	return ret;
}

vector< string > Msg::getSrcFieldsOnE2() const
{
	vector< pair< BindIndex, FuncId > > ids;
	vector< string > ret;

	e2_->getFieldsOfOutgoingMsg( mid_, ids );

	for ( unsigned int i = 0; i < ids.size(); ++i ) {
		string name = e2_->cinfo()->srcFinfoName( ids[i].first );
		if ( name == "" ) {
			cout << "Error: Msg::getSrcFieldsOnE2: Failed to find field on msg " <<
			e1_->getName() << "-->" << e2_->getName() << endl;
		} else {
			ret.push_back( name );
		}
	}
	return ret;
}

vector< string > Msg::getDestFieldsOnE1() const
{
	vector< pair< BindIndex, FuncId > > ids;
	vector< string > ret;

	e2_->getFieldsOfOutgoingMsg( mid_, ids );

	for ( unsigned int i = 0; i < ids.size(); ++i ) {
		string name = e1_->cinfo()->destFinfoName( ids[i].second );
		if ( name == "" ) {
			cout << "Error: Msg::getDestFieldsOnE1: Failed to find field on msg " <<
			e1_->getName() << "-->" << e2_->getName() << endl;
		} else {
			ret.push_back( name );
		}
	}
	return ret;
}

ObjId Msg::getAdjacent(ObjId obj) const
{
    return findOtherEnd(obj);
}

///////////////////////////////////////////////////////////////////////////
// Here we set up the Element related stuff for Msgs.
///////////////////////////////////////////////////////////////////////////

const Cinfo* Msg::initCinfo()
{
	///////////////////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////////////////
	static ReadOnlyValueFinfo< Msg, Id > e1(
		"e1",
		"Id of source Element.",
		&Msg::getE1
	);
	static ReadOnlyValueFinfo< Msg, Id > e2(
		"e2",
		"Id of source Element.",
		&Msg::getE2
	);

	static ReadOnlyValueFinfo< Msg, vector< string > > srcFieldsOnE1(
		"srcFieldsOnE1",
		"Names of SrcFinfos for messages going from e1 to e2. There are"
		"matching entries in the destFieldsOnE2 vector",
		&Msg::getSrcFieldsOnE1
	);
	static ReadOnlyValueFinfo< Msg, vector< string > > destFieldsOnE2(
		"destFieldsOnE2",
		"Names of DestFinfos for messages going from e1 to e2. There are"
		"matching entries in the srcFieldsOnE1 vector",
		&Msg::getDestFieldsOnE2
	);
	static ReadOnlyValueFinfo< Msg, vector< string > > srcFieldsOnE2(
		"srcFieldsOnE2",
		"Names of SrcFinfos for messages going from e2 to e1. There are"
		"matching entries in the destFieldsOnE1 vector",
		&Msg::getSrcFieldsOnE2
	);
	static ReadOnlyValueFinfo< Msg, vector< string > > destFieldsOnE1(
		"destFieldsOnE1",
		"Names of destFinfos for messages going from e2 to e1. There are"
		"matching entries in the srcFieldsOnE2 vector",
		&Msg::getDestFieldsOnE1
	);

        static ReadOnlyLookupValueFinfo< Msg, ObjId, ObjId > adjacent(
            "adjacent",
            "The element adjacent to the specified element",
            &Msg::getAdjacent);

	static Finfo* msgFinfos[] = {
		&e1,		// readonly value
		&e2,		// readonly value
		&srcFieldsOnE1,	// readonly value
		&destFieldsOnE2,	// readonly value
		&srcFieldsOnE2,	// readonly value
		&destFieldsOnE1,	// readonly value
                &adjacent, // readonly lookup value
	};

	static Cinfo msgCinfo (
		"Msg",	// name
		Neutral::initCinfo(),				// base class
		msgFinfos,
		sizeof( msgFinfos ) / sizeof( Finfo* ),	// num Fields
		0
		// new Dinfo< Msg >()
	);

	return &msgCinfo;
}

static const Cinfo* msgCinfo = Msg::initCinfo();

// Static func
void Msg::initMsgManagers()
{
	Dinfo< short > dummyDinfo;

	// This is to be the parent of all the msg managers.
	msgManagerId_ = Id::nextId();
	new Element( msgManagerId_, Neutral::initCinfo(), "Msgs", 1, 1 );

	SingleMsg::managerId_ = Id::nextId();
	new Element( SingleMsg::managerId_, SingleMsg::initCinfo(), 
		"singleMsg", 1, true );
	msgMgrs.push_back( SingleMsg::managerId_ );

	OneToOneMsg::managerId_ = Id::nextId();
	new Element( OneToOneMsg::managerId_, OneToOneMsg::initCinfo(),
		"oneToOneMsg", 1, true );
	msgMgrs.push_back( OneToOneMsg::managerId_ );

	OneToAllMsg::managerId_ = Id::nextId();
	new Element( OneToAllMsg::managerId_, OneToAllMsg::initCinfo(), 
		"oneToAllMsg", 1, true );
	msgMgrs.push_back( OneToAllMsg::managerId_ );

	DiagonalMsg::managerId_ = Id::nextId();
	new Element( DiagonalMsg::managerId_, DiagonalMsg::initCinfo(), 
		"diagonalMsg", 1, true );
	msgMgrs.push_back( DiagonalMsg::managerId_ );

	SparseMsg::managerId_ = Id::nextId();
	new Element( SparseMsg::managerId_, SparseMsg::initCinfo(), 
		"sparseMsg", 1, true );
	msgMgrs.push_back( SparseMsg::managerId_ );

	msgMgrs.push_back( msgManagerId_ );

	// Do the 'adopt' only after all the message managers exist - we need
	// the OneToAll manager for the adoption messages themselves.
	Shell::adopt( Id(), msgManagerId_ );
	Shell::adopt( msgManagerId_, SingleMsg::managerId_ );
	Shell::adopt( msgManagerId_, OneToOneMsg::managerId_ );
	Shell::adopt( msgManagerId_, OneToAllMsg::managerId_ );
	Shell::adopt( msgManagerId_, DiagonalMsg::managerId_ );
	Shell::adopt( msgManagerId_, SparseMsg::managerId_ );
}

void destroyMsgManagers()
{
	for ( vector< Id >::iterator i = msgMgrs.begin(); i != msgMgrs.end(); ++i)
		i->destroy();
}

// Static utility func used by derived classes to clean up their msgs
// Although all use this function we can't put it in the parent ~Msg
// function because C++ doesn't allow virtual class functions to be 
// called during the destructor.
void Msg::destroyDerivedMsg( Id managerId, MsgId mid )
{
	// This test is used because it is possible for the deletion order
	// when cleaning out the simulation to get rid of the msg Manager
	// before other things, which may have messages still to remove.
		/*
	if ( managerId.element() && managerId.element()->cinfo() ) {
		MsgDataHandler* mdh = dynamic_cast< MsgDataHandler* >(
			managerId.element()->dataHandler() );
		assert( mdh );
		mdh->dropMid( mid );
	}
	*/
}
