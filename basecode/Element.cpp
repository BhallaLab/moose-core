/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "FuncOrder.h"

Element::Element( Id id, const Cinfo* c, const string& name )
	:	name_( name ),
		id_( id ),
		cinfo_( c ), 
		msgBinding_( c->numBindIndex() ),
		msgDigest_( c->numBindIndex() ),
		isRewired_( false ),
		isDoomed_( false )
{
	id.bindIdToElement( this );
}


Element::~Element()
{
	// A flag that the Element is doomed, used to avoid lookups 
	// when deleting Msgs.
	id_.zeroOut();
	markAsDoomed();
	for ( vector< vector< MsgFuncBinding > >::iterator 
		i = msgBinding_.begin(); i != msgBinding_.end(); ++i ) {
		for ( vector< MsgFuncBinding >::iterator 
			j = i->begin(); j != i->end(); ++j ) {
			// This call internally protects against double deletion.
			Msg::deleteMsg( j->mid );
		}
	}

	for ( vector< ObjId >::iterator i = m_.begin(); i != m_.end(); ++i )
		Msg::deleteMsg( *i );
}

/////////////////////////////////////////////////////////////////////////
// Element info functions
/////////////////////////////////////////////////////////////////////////

const string& Element::getName() const
{
	return name_;
}

void Element::setName( const string& val )
{
	name_ = val;
}

const Cinfo* Element::cinfo() const
{
	return cinfo_;
}

Id Element::id() const
{
	return id_;
}

/////////////////////////////////////////////////////////////////////////
// Msg Management
/////////////////////////////////////////////////////////////////////////
void Element::addMsg( ObjId m )
{
	while ( m_.size() > 0 ) {
		if ( m_.back() == ObjId() )
			m_.pop_back();
		else
			break;
	}
	m_.push_back( m );
	markRewired();
}

class matchMid
{
	public:
		matchMid( ObjId mid )
			: mid_( mid )
		{;}

		bool operator()( const MsgFuncBinding& m ) const {
			return m.mid == mid_;
		}
	private:
		ObjId mid_;
};

/**
 * Called from ~Msg. This requires the usual scan through all msgs,
 * and could get inefficient.
 */
void Element::dropMsg( ObjId mid )
{
	if ( isDoomed() ) // This is a flag that the Element is doomed.
		return;
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), mid ), m_.end() );

	for ( vector< vector< MsgFuncBinding > >::iterator i = msgBinding_.begin(); i != msgBinding_.end(); ++i ) {
		matchMid match( mid ); 
		i->erase( remove_if( i->begin(), i->end(), match ), i->end() );
	}
	markRewired();
}

void Element::addMsgAndFunc( ObjId mid, FuncId fid, BindIndex bindIndex )
{
	if ( msgBinding_.size() < bindIndex + 1U )
		msgBinding_.resize( bindIndex + 1 );
	msgBinding_[ bindIndex ].push_back( MsgFuncBinding( mid, fid ) );
	markRewired();
}

void Element::clearBinding( BindIndex b )
{
	assert( b < msgBinding_.size() );
	vector< MsgFuncBinding > temp = msgBinding_[ b ];
	msgBinding_[ b ].resize( 0 );
	for( vector< MsgFuncBinding >::iterator i = temp.begin(); 
		i != temp.end(); ++i ) {
		Msg::deleteMsg( i->mid );
	}
	markRewired();
}

/// Used upon ending of MOOSE session, to rapidly clear out messages
void Element::clearAllMsgs()
{
	markAsDoomed();
	m_.clear();
	msgBinding_.clear();
	msgDigest_.clear();
}
/////////////////////////////////////////////////////////////////////////
// Msg Information
/////////////////////////////////////////////////////////////////////////

const vector< MsgDigest >& Element::msgDigest( unsigned int index )
{
	assert( index < msgDigest_.size() );
	if ( isRewired_ ) {
			digestMessages();
		isRewired_ = false;
	}
	return msgDigest_[ index ];
}

const vector< MsgFuncBinding >* Element::getMsgAndFunc( BindIndex b ) const
{
	if ( b < msgBinding_.size() )
		return &( msgBinding_[ b ] );
	return 0;
}

bool Element::hasMsgs( BindIndex b ) const
{
	return ( b < msgBinding_.size() && msgBinding_[b].size() > 0 );
}


void Element::showMsg() const
{
	cout << "Outgoing: \n";
	for ( map< string, Finfo* >::const_iterator i = 
		cinfo_->finfoMap().begin();
		i != cinfo_->finfoMap().end(); ++i ) {
		const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( i->second );
		if ( sf && msgBinding_.size() > sf->getBindIndex() ) {
			const vector< MsgFuncBinding >& mb = msgBinding_[ sf->getBindIndex()];
			unsigned int numTgt = mb.size();
			if ( numTgt > 0 ) {
				for ( unsigned int j = 0; j < numTgt; ++j ) {
					cout << sf->name() << " bindId=" << sf->getBindIndex() << ": ";
					cout << j << ": MessageId=" << mb[j].mid << 
					", FuncId=" << mb[j].fid << 
					", " << Msg::getMsg( mb[j].mid )->e1()->getName() << 
					" -> " <<
					Msg::getMsg( mb[j].mid )->e2()->getName() << endl;
				}
			}
		}
	}
	cout << "Dest and Src: \n";
	for ( unsigned int i = 0; i < m_.size(); ++i ) {
		const Msg* m = Msg::getMsg( m_[i] );
		cout << i << ": MessageId= " << m_[i] << 
			", e1= " << m->e1()->name_ <<
			", e2= " << m->e2()->name_ << endl;
	}
}


ObjId Element::findCaller( FuncId fid ) const
{
	for ( vector< ObjId >::const_iterator i = m_.begin(); 
		i != m_.end(); ++i )
	{
		const Msg* m = Msg::getMsg( *i );
		const Element* src;
		if ( m->e1() == this ) {
			src = m->e2();
		} else {
			src = m->e1();
		}
		unsigned int ret = src->findBinding( MsgFuncBinding( *i, fid ) );
		if ( ret != ~0U ) {
			return *i;
		}
	}
	return ObjId();
}

unsigned int Element::findBinding( MsgFuncBinding b ) const
{
	for ( unsigned int i = 0; i < msgBinding_.size(); ++i ) 
	{
		const vector< MsgFuncBinding >& mb = msgBinding_[i];
		vector< MsgFuncBinding>::const_iterator bi = 
			find( mb.begin(), mb.end(), b );
		if ( bi != mb.end() )
			return i;
	}
	return ~0;
}

const vector< ObjId >& Element::msgIn() const
{
	return m_;
}

vector< FuncOrder>  putFuncsInOrder( 
				const Element* elm, const vector< MsgFuncBinding >& vec )
{
	vector< FuncOrder > fo( vec.size() );
	for ( unsigned int j = 0; j < vec.size(); ++j ) {
		const MsgFuncBinding& mfb = vec[j];
		const Msg* msg = Msg::getMsg( mfb.mid );
		if ( msg->e1() == elm ) {
			fo[j].set( msg->e2()->cinfo()->getOpFunc( mfb.fid ), j );
		} else {
			fo[j].set( msg->e1()->cinfo()->getOpFunc( mfb.fid ), j );
		}
	}
	sort( fo.begin(), fo.end() );
	return fo;
}

void Element::putTargetsInDigest( 
				unsigned int srcNum, const MsgFuncBinding& mfb, 
				const FuncOrder& fo )
{
	const Msg* msg = Msg::getMsg( mfb.mid );
	vector< vector < Eref > > erefs;
	if ( msg->e1() == this )
		msg->targets( erefs );
	else if ( msg->e2() == this )
		msg->sources( erefs );
	else
		assert( 0 );
	for ( unsigned int j = 0; j < numData(); ++j ) {
		vector< MsgDigest >& md = 
			msgDigest_[ msgBinding_.size() * j + srcNum ];
		// k->func(); erefs[ j ];
		if ( md.size() == 0 || md.back().func != fo.func() ) {
			md.push_back( MsgDigest( fo.func(), erefs[j] ) );
		} else {
			md.back().targets.insert( md.back().targets.end(), 
					erefs[ j ].begin(),
					erefs[ j ].end() );
		}
	}
}

void Element::digestMessages()
{
	msgDigest_.clear();
	msgDigest_.resize( msgBinding_.size() * numData() );
	for ( unsigned int i = 0; i < msgBinding_.size(); ++i ) {
		// Go through and identify functions with the same ptr.
		vector< FuncOrder > fo = putFuncsInOrder( this, msgBinding_[i] );
		for ( vector< FuncOrder >::const_iterator 
						k = fo.begin(); k != fo.end(); ++k ) {
			const MsgFuncBinding& mfb = msgBinding_[i][ k->index() ];
			putTargetsInDigest( i, mfb, *k );
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Field Information
/////////////////////////////////////////////////////////////////////////
void Element::showFields() const
{
	vector< const SrcFinfo* > srcVec;
	vector< const DestFinfo* > destVec;
	vector< const SharedFinfo* > sharedVec;
	vector< const Finfo* > valueVec; // ValueFinfos are what is left.
	for ( map< string, Finfo* >::const_iterator i = 
		cinfo_->finfoMap().begin();
		i != cinfo_->finfoMap().end(); ++i ) {
		const SrcFinfo* sf = dynamic_cast< const SrcFinfo* >( i->second);
		const DestFinfo* df = dynamic_cast< const DestFinfo* >( i->second);
		const SharedFinfo* shf = dynamic_cast< const SharedFinfo* >( i->second);
		if ( sf )
			srcVec.push_back( sf );
		else if ( df )
			destVec.push_back( df );
		else if ( shf )
			sharedVec.push_back( shf );
		else
			valueVec.push_back( i->second );
	}

	cout << "Showing SrcFinfos: \n";
	for ( unsigned int i = 0; i < srcVec.size(); ++i )
		cout << i << ": " << srcVec[i]->name() << "	Bind=" << srcVec[i]->getBindIndex() << endl;
	cout << "Showing " << destVec.size() << " DestFinfos: \n";
	/*
	for ( unsigned int i = 0; i < destVec.size(); ++i )
		cout << i << ": " << destVec[i]->name() << "	FuncId=" << destVec[i]->getFid() << endl;
		*/
	cout << "Showing SharedFinfos: \n";
	for ( unsigned int i = 0; i < sharedVec.size(); ++i ) {
		cout << i << ": " << sharedVec[i]->name() << "	Src=[ ";
		for ( unsigned int j = 0; j < sharedVec[i]->src().size(); ++j )
			cout << " " << sharedVec[i]->src()[j]->name();
		cout << " ]	Dest=[ ";
		for ( unsigned int j = 0; j < sharedVec[i]->dest().size(); ++j )
			cout << " " << sharedVec[i]->dest()[j]->name();
		cout << " ]\n";
	}
	cout << "Listing ValueFinfos: \n";
	Eref er = this->id().eref();
	string val;
	for ( unsigned int i = 0; i < valueVec.size(); ++i ) {
			valueVec[i]->strGet( er, valueVec[i]->name(), val );
		cout << i << ": " << valueVec[i]->name() << "	" <<
			val << endl;
	}
}

void Element::destroyElementTree( const vector< Id >& tree )
{
	for( vector< Id >::const_iterator i = tree.begin(); 
		i != tree.end(); i++ )
		i->element()->markAsDoomed(); // Indicate that Element is doomed
	bool killShell = false;

	// Do not destroy the shell till the very end.
	for( vector< Id >::const_iterator i = tree.begin(); 
		i != tree.end(); i++ ) {
		if ( *i == Id() )
			killShell = true;
		else
			i->destroy();
	}
	if ( killShell )
		Id().destroy();
}

void Element::markAsDoomed()
{
	isDoomed_ = true;
}

bool Element::isDoomed() const
{
	return isDoomed_;
}

void Element::markRewired()
{
	isRewired_ = true;
}

void Element::zombieSwap( const Cinfo* newCinfo )
{
	cinfo_ = newCinfo;
	// Stuff to be done here for data.
}

//////////////////////////////////////////////////////////////////////////
// Message traversal
//////////////////////////////////////////////////////////////////////////
unsigned int Element::getOutputs( vector< Id >& ret, const SrcFinfo* finfo )
	const
{
	assert( finfo ); // would like to check that finfo is on this.
	unsigned int oldSize = ret.size();
	
	const vector< MsgFuncBinding >* msgVec =
		getMsgAndFunc( finfo->getBindIndex() );
	for ( unsigned int i = 0; i < msgVec->size(); ++i ) {
		const Msg* m = Msg::getMsg( (*msgVec)[i].mid );
		assert( m );
		Id id = m->e2()->id();
		if ( m->e2() == this )
			id = m->e1()->id();
		ret.push_back( id );
	}
	
	return ret.size() - oldSize;
}

unsigned int Element::getInputs( vector< Id >& ret, const DestFinfo* finfo )
	const
{
	assert( finfo ); // would like to check that finfo is on src.
	unsigned int oldSize = ret.size();
	
	FuncId fid = finfo->getFid();
	vector< ObjId > caller;
	getInputMsgs( caller, fid );
	for ( vector< ObjId >::iterator i = caller.begin(); 
		i != caller.end(); ++i  ) {
		const Msg* m = Msg::getMsg( *i );
		assert( m );

		Id id = m->e1()->id();
		if ( m->e1() == this )
			id = m->e2()->id();
		ret.push_back( id );
	}
	return ret.size() - oldSize;
}

unsigned int Element::getNeighbours( vector< Id >& ret, const Finfo* finfo )
	const
{
	assert( finfo );
	
	const SrcFinfo* srcF = dynamic_cast< const SrcFinfo* >( finfo );
	const DestFinfo* destF = dynamic_cast< const DestFinfo* >( finfo );
	const SharedFinfo* sharedF = dynamic_cast< const SharedFinfo* >( finfo );
	assert( srcF || destF || sharedF );

	ret.resize( 0 );
	
	if ( srcF )
		return getOutputs( ret, srcF );
	else if ( destF )
		return getInputs( ret, destF );
	else
		if ( ! sharedF->src().empty() )
			return getOutputs( ret, sharedF->src().front() );
		else if ( ! sharedF->dest().empty() ) {
			Finfo* subFinfo = sharedF->dest().front();
			const DestFinfo* subDestFinfo =
				dynamic_cast< const DestFinfo* >( subFinfo );
			assert( subDestFinfo );
			return getInputs( ret, subDestFinfo );
		} else {
			assert( 0 );
		}
	return 0;
}

// May return multiple Msgs.
unsigned int Element::getInputMsgs( vector< ObjId >& caller, FuncId fid)
	const
{
	for ( vector< ObjId >::const_iterator i = m_.begin(); 
		i != m_.end(); ++i )
	{
		const Msg* m = Msg::getMsg( *i );
		const Element* src;
		if ( m->e1() == this ) {
			src = m->e2();
		} else {
			src = m->e1();
		}
		unsigned int ret = src->findBinding( MsgFuncBinding( *i, fid ) );
		if ( ret != ~0U ) {
			caller.push_back( *i );
		}
	}
	return caller.size();
}

unsigned int Element::getFieldsOfOutgoingMsg( ObjId mid,
	vector< pair< BindIndex, FuncId > >& ret ) const
{
	ret.resize( 0 );
	for ( unsigned int i = 0; i < msgBinding_.size(); ++i )
	{
		const vector< MsgFuncBinding >& mb = msgBinding_[i];
		for ( vector< MsgFuncBinding >::const_iterator j = mb.begin();
			j != mb.end(); ++j ) {
			if ( j->mid == mid ) {
				ret.push_back( pair< BindIndex, FuncId >( i, j->fid ));
			}
		}
	}
	return ret.size();
}
