/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

Element::Element( Id id, const Cinfo* c, const string& name, 
	unsigned int numData, bool isGlobal )
	:	name_( name ),
		id_( id ),
		cinfo_( c ), 
		msgBinding_( c->numBindIndex() )
{
	id.bindIdToElement( this );
	data_ = c->dinfo()->allocData( numData );
	c->postCreationFunc( id, this );
}


/*
 * Used for copies. Note that it does NOT call the postCreation Func,
 * so FieldElements are copied rather than created by the Cinfo when
 * the parent element is created. This allows the copied FieldElements to
 * retain info from the originals.
 */
Element::Element( Id id, const Element* orig, unsigned int n,
	bool toGlobal)
	:	name_( orig->getName() ),
		id_( id ),
		cinfo_( orig->cinfo_ ), 
		msgBinding_( orig->cinfo_->numBindIndex() )
{
	if ( n >= 1 ) {
		data_ = cinfo()->dinfo()->copyData( orig->data_, orig->numData_, n);
	}
	id.bindIdToElement( this );
	// cinfo_->postCreationFunc( id, this );
}

Element::~Element()
{
	// cout << "deleting element " << getName() << endl;
	cinfo_->dinfo()->destroyData( data_ );
	clearCinfoAndMsgs();
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

unsigned int Element::numData() const
{
	return numData_;
}

unsigned int Element::numField( unsigned int entry ) const
{
	return 1;
}


const Cinfo* Element::cinfo() const
{
	return cinfo_;
}

Id Element::id() const
{
	return id_;
}

char* Element::data( unsigned int rawIndex, unsigned int fieldIndex ) const
{
	assert( rawIndex < numData_ );
	return data_ + ( rawIndex * cinfo()->dinfo()->size() );
}

void Element::resize( unsigned int newNumData )
{
	char* temp = data_;
	data_ = cinfo()->dinfo()->copyData( temp, numData_, newNumData );
	cinfo()->dinfo()->destroyData( temp );
}

/////////////////////////////////////////////////////////////////////////
// Msg Management
/////////////////////////////////////////////////////////////////////////
void Element::addMsg( MsgId m )
{
	while ( m_.size() > 0 ) {
		if ( m_.back() == Msg::bad )
			m_.pop_back();
		else
			break;
	}
	m_.push_back( m );
}

class matchMid
{
	public:
		matchMid( MsgId mid )
			: mid_( mid )
		{;}

		bool operator()( const MsgFuncBinding& m ) const {
			return m.mid == mid_;
		}
	private:
		MsgId mid_;
};

/**
 * Called from ~Msg. This requires the usual scan through all msgs,
 * and could get inefficient.
 */
void Element::dropMsg( MsgId mid )
{
	if ( cinfo_ == 0 ) // Don't need to clear if Element itself is going.
		return;
	// Here we have the spectacularly ugly C++ erase-remove idiot.
	m_.erase( remove( m_.begin(), m_.end(), mid ), m_.end() );

	for ( vector< vector< MsgFuncBinding > >::iterator i = msgBinding_.begin(); i != msgBinding_.end(); ++i ) {
		matchMid match( mid ); 
		i->erase( remove_if( i->begin(), i->end(), match ), i->end() );
	}
}

void Element::addMsgAndFunc( MsgId mid, FuncId fid, BindIndex bindIndex )
{
	if ( msgBinding_.size() < bindIndex + 1U )
		msgBinding_.resize( bindIndex + 1 );
	msgBinding_[ bindIndex ].push_back( MsgFuncBinding( mid, fid ) );
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
}
/////////////////////////////////////////////////////////////////////////
// Msg Information
/////////////////////////////////////////////////////////////////////////

const vector< MsgDigest >& Element::msgDigest( unsigned int index ) const
{
	assert( index < msgDigest_.size() );
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
					cout << j << ": MsgId=" << mb[j].mid << 
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
		cout << i << ": MsgId= " << m_[i] << 
			", e1= " << m->e1()->name_ <<
			", e2= " << m->e2()->name_ << endl;
	}
}


MsgId Element::findCaller( FuncId fid ) const
{
	for ( vector< MsgId >::const_iterator i = m_.begin(); 
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
	return Msg::bad;
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

 const vector< MsgId >& Element::msgIn() const
 {
 	return m_;
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
		i->element()->cinfo_ = 0; // Indicate that Element is doomed
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
	vector< MsgId > caller;
	getInputMsgs( caller, fid );
	for ( vector< MsgId >::iterator i = caller.begin(); 
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
unsigned int Element::getInputMsgs( vector< MsgId >& caller, FuncId fid)
	const
{
	for ( vector< MsgId >::const_iterator i = m_.begin(); 
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

unsigned int Element::getFieldsOfOutgoingMsg( MsgId mid,
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

// Protected function, used only during Element destruction.
void Element::clearCinfoAndMsgs()
{
	cinfo_ = 0; // A flag that the Element is doomed, used to avoid lookups when deleting Msgs.
	for ( vector< vector< MsgFuncBinding > >::iterator i = msgBinding_.begin(); i != msgBinding_.end(); ++i ) {
		for ( vector< MsgFuncBinding >::iterator j = i->begin(); j != i->end(); ++j ) {
			// This call internally protects against double deletion.
			Msg::deleteMsg( j->mid );
		}
	}

	for ( vector< MsgId >::iterator i = m_.begin(); i != m_.end(); ++i )
		if ( *i ) // Dropped Msgs set this pointer to zero, so skip them.
			Msg::deleteMsg( *i );
}
