/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DataDimensions.h"
#include "AnyDimGlobalHandler.h"
#include "AnyDimHandler.h"
#include "DataHandlerWrapper.h"

/**
 * This version is used when making zombies. We want to have a
 * temporary Element for field access but nothing else, and it
 * should not mess with messages or Ids.
 */
Element::Element( Id id, const Cinfo* c, DataHandler* d )
	: id_( id ), cinfo_( c ), group_( 0 )
{
	dataHandler_ = new DataHandlerWrapper( d );
}

unsigned int numDimensionsActuallyUsed( 
	const vector< unsigned int >& dimensions )
{
	unsigned int ret = 0;
	for ( unsigned int i = 0; i < dimensions.size(); ++i ) {
		if ( dimensions[i] > 1 ) {
			++ret;
		} else {
			break; // Do not permit high dimensions later.
		}
	}
	return ret;
}

Element::Element( Id id, const Cinfo* c, const string& name, 
	const vector< unsigned int >& dimensions, bool isGlobal )
	:	name_( name ),
		id_( id ),
		cinfo_( c ), 
		group_( 0 ),
		msgBinding_( c->numBindIndex() )
{
	unsigned int numRealDimensions = numDimensionsActuallyUsed( dimensions);

	if ( numRealDimensions == 0 ) {
		if ( isGlobal )
			dataHandler_ = new ZeroDimGlobalHandler( c->dinfo() );
		else
			dataHandler_ = new ZeroDimHandler( c->dinfo() );
	} else if ( numRealDimensions == 1 ) {
		if ( isGlobal )
			dataHandler_ = new OneDimGlobalHandler( c->dinfo() );
		else
			dataHandler_ = new OneDimHandler( c->dinfo() );	
	} else {
		if ( isGlobal )
			dataHandler_ = new AnyDimGlobalHandler( c->dinfo() );
		else
			dataHandler_ = new AnyDimHandler( c->dinfo() );	
	}
	dataHandler_->resize( dimensions );

	id.bindIdToElement( this );
	c->postCreationFunc( id, this );
}

Element::Element( Id id, const Cinfo* c, const string& name, 
	DataHandler* dataHandler )
	:	name_( name ),
		id_( id ),
		dataHandler_( dataHandler ),
		cinfo_( c ), 
		group_( 0 ),
		msgBinding_( c->numBindIndex() )
{
	id.bindIdToElement( this );
	c->postCreationFunc( id, this );
}

/*
 * Used for copies. Note that it does NOT call the postCreation Func,
 * so FieldElements are copied rather than created by the Cinfo when
 * the parent element is created. This allows the copied FieldElements to
 * retain info from the originals.
 */
Element::Element( Id id, const Element* orig, unsigned int n, bool toGlobal)
	:	name_( orig->getName() ),
		id_( id ),
		cinfo_( orig->cinfo_ ), 
		group_( orig->group_ ),
		msgBinding_( orig->cinfo_->numBindIndex() )
{
	if ( n <= 1 ) {
		dataHandler_ = orig->dataHandler_->copy( toGlobal );
	} else {
		dataHandler_ = orig->dataHandler_->copyToNewDim( n, toGlobal  );
	}
	id.bindIdToElement( this );
	// cinfo_->postCreationFunc( id, this );
}

Element::~Element()
{
	// cout << "deleting element " << getName() << endl;
	delete dataHandler_;
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

const string& Element::getName() const
{
	return name_;
}

void Element::setName( const string& val )
{
	name_ = val;
}

unsigned int Element::getGroup() const
{
	return group_;
}

void Element::setGroup( unsigned int val )
{
	group_ = val;
}

/**
 * The indices handled by each thread are in blocks
 * Thread0 handles the first (numData_ / numThreads ) indices
 * Thread1 handles ( numData_ / numThreads ) to (numData_*2 / numThreads)
 * and so on.
 */
void Element::process( const ProcInfo* p, FuncId fid )
{
	dataHandler_->process( p, this, fid );
}


void Element::addMsg( MsgId m )
{
	while ( m_.size() > 0 ) {
		if ( m_.back() == Msg::badMsg )
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

const vector< MsgFuncBinding >* Element::getMsgAndFunc( BindIndex b ) const
{
	if ( b < msgBinding_.size() )
		return &( msgBinding_[ b ] );
	return 0;
}


const Cinfo* Element::cinfo() const
{
	return cinfo_;
}

DataHandler* Element::dataHandler() const
{
	return dataHandler_;
}

/**
* Resizes the current data, may include changing dimensions.
* Returns the new dataHandler if needed, and NULL on failure.
* When resizing it uses the current data and puts it treadmill-
* fashion into the new dimensions. This means that if we had a
* 2-D array and add a z dimension while keeping x and y fixed, we
* should just repeat the same plane of data for all z values.
* But it will get terribly messy if we change x and y dimensions.
* Note that the resizing only works on the data dimensions, it
* does not touch the field dimensions.
*/
DataHandler* Element::resize( const vector< unsigned int >& dims )
{
	unsigned int numRealDimensions = numDimensionsActuallyUsed( dims );
	if ( numRealDimensions == dataHandler_->numDimensions() ) {
		dataHandler_->resize( dims );
		return dataHandler_;
	} else {
		DataHandler* old = dataHandler_;
		if ( numRealDimensions == 0 ) {
			if ( old->isGlobal() )
				dataHandler_ = new ZeroDimGlobalHandler( old->dinfo() );
			else
				dataHandler_ = new ZeroDimHandler( old->dinfo() );
		} else if ( numRealDimensions == 1 ) {
			if ( old->isGlobal() )
				dataHandler_ = new OneDimGlobalHandler( old->dinfo() );
			else
				dataHandler_ = new OneDimHandler( old->dinfo() );	
		} else {
			if ( old->isGlobal() )
				dataHandler_ = new AnyDimGlobalHandler( old->dinfo() );
			else
				dataHandler_ = new AnyDimHandler( old->dinfo() );	
		}
		dataHandler_->resize( dims );
		unsigned int oldSize = old->localEntries();
		unsigned int newSize = dataHandler_->localEntries();
		const char* data = *( old->begin() );

		unsigned int start = dataHandler_->begin().index().data(); 
		for( unsigned int i = 0; i < newSize; i += oldSize ) {
			dataHandler_->setDataBlock( data, oldSize, DataId( i + start ));
		}
		delete old;
	}
	return dataHandler_;
}

Id Element::id() const
{
	return id_;
}

/**
 * Executes a queue entry from the buffer.
 */
void Element::exec( const Qinfo* qi, const double* arg )
	const
{
	static const unsigned int ObjFidSizeInDoubles = 
		1 + ( sizeof( ObjFid ) - 1 ) / sizeof( double );
	if ( qi->isDirect() ) { // Direct Q entry, where the first part
		// of Data specifies the target Element.
		const ObjFid *ofid = reinterpret_cast< const ObjFid* >( arg );
		const OpFunc* f = 
			ofid->oi.element()->cinfo()->getOpFunc( ofid->fid );
		if ( ofid->oi.dataId == DataId::bad() ) return;
		if ( ofid->oi.dataId == DataId::any() ) {
			// Here we iterate through the DataId, using the
			// numEntries and entrySize of the ofid to set args.
		} else {
			f->op( ofid->oi.eref(), qi, arg + ObjFidSizeInDoubles );
		}
	} else {
		assert( qi->bindIndex() < msgBinding_.size() );
		vector< MsgFuncBinding >::const_iterator end = 
			msgBinding_[ qi->bindIndex() ].end();
		for ( vector< MsgFuncBinding >::const_iterator i =
			msgBinding_[ qi->bindIndex() ].begin(); i != end; ++i ) {
			assert( i->mid != 0 );
			assert( Msg::getMsg( i->mid ) != 0 );
			Msg::getMsg( i->mid )->exec( qi, arg, i->fid );
		}
	}
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
	return Msg::badMsg;
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


void Element::destroyElementTree( const vector< Id >& tree )
{
	for( vector< Id >::const_iterator i = tree.begin(); 
		i != tree.end(); i++ )
		i->operator()()->cinfo_ = 0; // Indicate that Element is doomed
	for( vector< Id >::const_iterator i = tree.begin(); 
		i != tree.end(); i++ )
		i->destroy();
		// delete i->operator()();
}

void Element::zombieSwap( const Cinfo* newCinfo, DataHandler* newDataHandler )
{
	cinfo_ = newCinfo;
	delete dataHandler_;
	dataHandler_ = newDataHandler;
}

unsigned int Element::getOutputs( vector< Id >& ret, const SrcFinfo* finfo )
	const
{
	assert( finfo ); // would like to check that finfo is on this.
	const vector< MsgFuncBinding >* msgVec =
		getMsgAndFunc( finfo->getBindIndex() );
	ret.resize( 0 );
	for ( unsigned int i = 0; i < msgVec->size(); ++i ) {
		const Msg* m = Msg::getMsg( (*msgVec)[i].mid );
		assert( m );
		Id id = m->e2()->id();
		if ( m->e2() == this )
			id = m->e1()->id();
		ret.push_back( id );
	}
	return ret.size();
}

unsigned int Element::getInputs( vector< Id >& ret, const DestFinfo* finfo )
	const
{
	assert( finfo ); // would like to check that finfo is on src.
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
	return ret.size();
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
