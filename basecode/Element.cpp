/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "DataHandlerWrapper.h"

/**
 * This version is used when making zombies. We want to have a
 * temporary Element for field access but nothing else, and it
 * should not mess with messages or Ids.
 */
Element::Element( const Cinfo* c, DataHandler* d )
{
	dataHandler_ = new DataHandlerWrapper( d );
	cinfo_ = c;
}

Element::Element( Id id, const Cinfo* c, const string& name, 
	const vector< unsigned int >& dimensions, bool isGlobal )
	:	name_( name ),
		id_( id ),
		sendBuf_( 0 ), 
		cinfo_( c ), 
		msgBinding_( c->numBindIndex() )
{
	unsigned int numRealDimensions = 0;

	for ( unsigned int i = 0; i < dimensions.size(); ++i ) {
		if ( dimensions[i] > 1 ) {
			++numRealDimensions;
		} else {
			break; // Do not permit high dimensions later.
		}
	}

	if ( numRealDimensions == 0 ) {
		if ( isGlobal )
			dataHandler_ = new ZeroDimGlobalHandler( c->dinfo() );
		else
			dataHandler_ = new ZeroDimHandler( c->dinfo() );
		dataHandler_->allocate();
	} else if ( numRealDimensions == 1 ) {
		if ( isGlobal )
			dataHandler_ = new OneDimGlobalHandler( c->dinfo() );
		else
			dataHandler_ = new OneDimHandler( c->dinfo() );	
		dataHandler_->setNumData1( dimensions[ 0 ] );
	} else {
		cout << "Don't yet have Two or higher DimHandler\n";
		exit( 0 );
	}

	id.bindIdToElement( this );
	c->postCreationFunc( id, this );
}

Element::Element( Id id, const Cinfo* c, const string& name, 
	DataHandler* dataHandler )
	:	name_( name ),
		id_( id ),
		dataHandler_( dataHandler ),
		sendBuf_( 0 ), 
		cinfo_( c ), 
		msgBinding_( c->numBindIndex() )
{
	id.bindIdToElement( this );
	c->postCreationFunc( id, this );
}

Element::Element( Id id, const Element* orig, unsigned int n )
	:	name_( orig->getName() ),
		id_( id ),
		dataHandler_( orig->dataHandler_->copy( 
			n, orig->dataHandler_->isGlobal() ) ),
		sendBuf_( 0 ), 
		cinfo_( orig->cinfo_ ), 
		msgBinding_( orig->cinfo_->numBindIndex() )
{
	id.bindIdToElement( this );
	cinfo_->postCreationFunc( id, this );
}

Element::~Element()
{
	delete[] sendBuf_;
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

/**
 * The indices handled by each thread are in blocks
 * Thread0 handles the first (numData_ / numThreads ) indices
 * Thread1 handles ( numData_ / numThreads ) to (numData_*2 / numThreads)
 * and so on.
 */
void Element::process( const ProcInfo* p )
{
	dataHandler_->process( p, this );
	/*
	char* data = d_;
	unsigned int start =
		( numData_ * p->threadIndexInGroup ) / p->numThreadsInGroup;
	unsigned int end =
		( numData_ * ( p->threadIndexInGroup + 1) ) / p->numThreadsInGroup;
	data += start * dataSize_;
	for ( unsigned int i = start; i < end; ++i ) {
		reinterpret_cast< Data* >( data )->process( p, Eref( this, i ) );
		data += dataSize_;
	}
	*/
}

double Element::sumBuf( SyncId slot, unsigned int i ) const
{
	vector< unsigned int >::const_iterator offset = 
		procBufRange_.begin() + slot + i * numRecvSlots_;
	vector< double* >::const_iterator begin = 
		procBuf_.begin() + *offset++;
	vector< double* >::const_iterator end = 
		procBuf_.begin() + *offset;
	double ret = 0.0;
	for ( vector< double* >::const_iterator i = begin; 
		i != end; ++i )
		ret += **i;
	return ret;
}

double Element::prdBuf( SyncId slot, unsigned int i, double v )
	const
{
	vector< unsigned int >::const_iterator offset = 
		procBufRange_.begin() + slot + i * numRecvSlots_;
	vector< double* >::const_iterator begin = 
		procBuf_.begin() + *offset++;
	vector< double* >::const_iterator end = 
		procBuf_.begin() + *offset;
	for ( vector< double* >::const_iterator i = begin;
		i != end; ++i )
		v *= **i;
	return v;
}

double Element::oneBuf( SyncId slot, unsigned int i ) const
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot + i * numRecvSlots_;
	assert( offset + 1 < procBufRange_.size() );
	return *procBuf_[ procBufRange_[ offset ] ];
}

double* Element::getBufPtr( SyncId slot, unsigned int i )
{
	// unsigned int offset = i * numData_ + slot;
	unsigned int offset = slot + i * numRecvSlots_;
	assert( offset + 1 < procBufRange_.size() );
	return procBuf_[ procBufRange_[ offset ] ];
}

void Element::ssend1( SyncId slot, unsigned int i, double v )
{
	sendBuf_[ slot + i * numSendSlots_ ] = v;
}

void Element::ssend2( SyncId slot, unsigned int i, double v1, double v2 )
{
	double* sb = sendBuf_ + slot + i * numSendSlots_;
	*sb++ = v1;
	*sb = v2;
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

Id Element::id() const
{
	return id_;
}

/*
 * Asynchronous send
 * At this point the Qinfo has the values for 
 * Eref::index() and size assigned.
 * ProcInfo is used only for p->qId?
 * Qinfo needs to have funcid put in, and Msg Id.
 * Msg info is also need to work out if Q entry is forward or back.
 */
void Element::asend( Qinfo& q, BindIndex bindIndex, 
	const ProcInfo *p, const char* arg ) const
{
	assert ( bindIndex < msgBinding_.size() );
	vector< MsgFuncBinding >::const_iterator end = 
		msgBinding_[ bindIndex ].end();
	for ( vector< MsgFuncBinding >::const_iterator i =
		msgBinding_[ bindIndex ].begin(); i != end; ++i ) {
		q.assembleOntoQ( *i, this, p, arg );
		/*
		const Msg* m = Msg::getMsg( i->mid );
		q.setForward( m->isForward( this ) );
		if ( m->isMsgHere( q ) ) {
			q.assignQblock( m, p );
			q.addToQ( p->threadId, *i, arg );
		}
		*/
	}
}

/*
 * Asynchronous send to specific target.
 * Scan through potential targets, figure out direction, 
 * copy over FullId to sit in specially assigned space on queue.
 *
 * This may seem easier to do if we don't even bother with the Msg,
 * and just plug in the queue entry.
 * but there is a requirement that all function calls should be able
 * to trace back their calling Element. At present that goes by the Msg.
 *
 */
void Element::tsend( Qinfo& q, BindIndex bindIndex, 
	const ProcInfo *p, const char* arg, const FullId& target ) const
{
	assert ( bindIndex < msgBinding_.size() );
	Element *e = target.id();
	for ( vector< MsgFuncBinding >::const_iterator i =
		msgBinding_[ bindIndex ].begin(); 
		i != msgBinding_[ bindIndex ].end(); ++i ) {
		const Msg* m = Msg::getMsg( i->mid );
		q.setForward( m->isForward( this ) );
		if ( q.isForward() ) {
			if ( m->e2() == e && m->isMsgHere( q ) ) {
				q.assignQblock( m, p );
				q.addSpecificTargetToQ( p->threadId, *i, arg, target.dataId );
				return;
			}
		} else {
			if ( m->e1() == e && m->isMsgHere( q ) ) {
				q.assignQblock( m, p );
				q.addSpecificTargetToQ( p->threadId, *i, arg, target.dataId );
				return;
			}
		}
	}
	cout << "Warning: Element::tsend: Failed to find specific target " <<
		target << endl;
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
					", " << Msg::getMsg( mb[j].mid )->e1() << " -> " <<
					Msg::getMsg( mb[j].mid )->e2() << endl;
				}
			}
		}
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
	cout << "Showing DestFinfos: \n";
	for ( unsigned int i = 0; i < destVec.size(); ++i )
		cout << i << ": " << destVec[i]->name() << "	FuncId=" << destVec[i]->getFid() << endl;
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
	for ( unsigned int i = 0; i < valueVec.size(); ++i )
		cout << i << ": " << valueVec[i]->name() << endl;
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
	// DataHandler* oldDataHandler = dataHandler_;
	dataHandler_ = newDataHandler;
	// delete oldDataHandler_;
}

