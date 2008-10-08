/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifdef USE_MPI
#include "moose.h"
#include <mpi.h>
#include "PostMaster.h"
#include "ProxyElement.h"
#include "AsyncDestFinfo.h"
#include <sstream>
// #include <typeinfo>
// #include "../element/Neutral.h" // Possibly only need this for testing.
// #include "../shell/Shell.h" // Possibly only need this for testing.

#define DATA_TAG 0

const Cinfo* initPostMasterCinfo()
{
	/**
	 * This shared message communicates between the ParTick and 
	 * the PostMaster
	 */
	static Finfo* parShared[] = 
	{
		// This first entry is to tell the PostMaster to post iRecvs
		// The argument is the ordinal number of the clock tick
		new DestFinfo( "postIrecv", Ftype1< int >::global(), 
			RFCAST( &PostMaster::postIrecv ) ),
		// The second entry is to tell the PostMaster to post 'send'
		new DestFinfo( "postSend", Ftype1< int >::global(), 
			RFCAST( &PostMaster::postSend ) ),
		// The third entry is for polling the receipt of incoming data.
		// Each PostMaster does an MPI_Test on the earlier posted iRecv.
		new DestFinfo( "poll", Ftype1< int >::global(), 
			RFCAST( &PostMaster::poll ) ),
		// The fourth entry is for harvesting the poll request.
		// The argument is the node number handled by the postmaster.
		// It comes back when the polling on that postmaster is done.
		new SrcFinfo( "harvestPoll", Ftype1< unsigned int >::global() )
	};

	static Finfo* serialShared[] =
	{
		new SrcFinfo( "rawAdd", // addmsg using serialized data.
			Ftype1< string >::global() ),
		new SrcFinfo( "rawCopy", // copy using serialized data.
			Ftype1< string >::global() ),
		new SrcFinfo( "rawTest", // copy using serialized data.
			Ftype1< string >::global() ),
	};

	static Finfo* postMasterFinfos[] = 
	{
		new ValueFinfo( "localNode", 
					ValueFtype1< unsigned int >::global(),
					GFCAST( &PostMaster::getMyNode ), &dummyFunc 
		),
		new ValueFinfo( "remoteNode", 
					ValueFtype1< unsigned int >::global(),
					GFCAST( &PostMaster::getRemoteNode ),
					RFCAST( &PostMaster::setRemoteNode )
		),

		// derived from DestFinfo, main difference is it accepts all
		// kinds of incoming data.
		new AsyncDestFinfo( "async", 
			Ftype2< char*, unsigned int >::global(), 
			RFCAST( &PostMaster::async ) ),

		////////////////////////////////////////////////////////////////
		//	Shared messages.
		////////////////////////////////////////////////////////////////
		new SharedFinfo( "parTick", parShared, 
			sizeof( parShared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "serial", serialShared, 
			sizeof( serialShared ) / sizeof( Finfo* ) ),
	};

	static Cinfo postMasterCinfo(
				"PostMaster",
				"Upi Bhalla",
				"PostMaster object. Manages parallel communications.",
				initNeutralCinfo(),
				postMasterFinfos,
				sizeof( postMasterFinfos ) / sizeof( Finfo* ),
				ValueFtype1< PostMaster >::global()
	);

	return &postMasterCinfo;
}

static const Cinfo* postMasterCinfo = initPostMasterCinfo();

static const Slot pollSlot = 
	initPostMasterCinfo()->getSlot( "parTick.harvestPoll" );
static const Slot addSlot = 
	initPostMasterCinfo()->getSlot( "serial.rawAdd" );
static const Slot copySlot = 
	initPostMasterCinfo()->getSlot( "serial.rawCopy" );
static const Slot testSlot = 
	initPostMasterCinfo()->getSlot( "serial.rawTest" );

//////////////////////////////////////////////////////////////////
// Here are the PostMaster class functions.
//////////////////////////////////////////////////////////////////
PostMaster::PostMaster()
	: remoteNode_( 0 ), 
	sendBuf_( 1000, 0 ), 
	sendBufPos_( 0 ), 
	recvBuf_( 1000, 0 ), 
	donePoll_( 0 ), comm_( &MPI_INTRA_COMM )
{
	localNode_ = MPI_INTRA_COMM.Get_rank(); 
	request_ = 0;
}

//////////////////////////////////////////////////////////////////
// Here we put the PostMaster Moose functions.
//////////////////////////////////////////////////////////////////

unsigned int PostMaster::getMyNode( Eref e )
{
		return static_cast< PostMaster* >( e.data() )->localNode_;
}

unsigned int PostMaster::getRemoteNode( Eref e )
{
		return static_cast< PostMaster* >( e.data() )->remoteNode_;
}

void PostMaster::setRemoteNode( const Conn* c, unsigned int node )
{
		static_cast< PostMaster* >( c->data() )->remoteNode_ = node;
}

/////////////////////////////////////////////////////////////////////
// Stuff data into async buffer. Used mostly for testing.
/////////////////////////////////////////////////////////////////////
void PostMaster::async( const Conn* c, char* data, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c->data() );
	assert( pm != 0 );
	void* addr = pm->innerGetAsyncParBuf( c, size );
	memcpy( addr, data, size );
}

/////////////////////////////////////////////////////////////////////
// Here we handle passing messages to off-nodes
/////////////////////////////////////////////////////////////////////

/*
// Just to help me remember how to use the typeid from RTTI.
// This will work only between identical compilers, I think.
// Something more exciting happens with shared finfos.
const char* ftype2str( const Ftype *f )
{
	return typeid( *f ).name();
}
*/


/////////////////////////////////////////////////////////////////////
// Utility function for accessing postmaster data buffer.
/////////////////////////////////////////////////////////////////////
/**
 * This function just passes the next free location over to the calling 
 * function. It does not store the targetIndex.
 * It internally increments the current location of the buffer.
 * If we don't use MPI, then this whole file is unlikely to be compiled.
 * So we define the dummy version of the function in DerivedFtype.cpp.
 */
#if 0
void* PostMaster::innerGetParBuf( 
				unsigned int targetIndex, unsigned int size )
{
	if ( size + outBufPos_ > outBufSize_ ) {
		cout << "in getParBuf: Out of space in outBuf.\n";
		// Do something clever here to send another installment
		return 0;
	}
	outBufPos_ += size;
	return static_cast< void* >( outBuf_ + outBufPos_ - size );
}

void PostMaster::parseMsgRequest( const char* req, Element* self )
{
	// sscanf( req, "%d %d %s", srcId, destId, typeSig );
	// string sreq( req );

	Id srcId;
	Id destId;
	string typeSig;
	string targetFname;
	unsigned int msgIndex;

	istringstream istr;
	istr.str( req );
	istr >> srcId >> destId >> typeSig >> targetFname 
						>> msgIndex >> ws;
	assert ( istr.eof() );
	assert ( !srcId.bad() );
	assert ( !destId.bad() );

	Element* dest = destId();
	assert( dest != 0 );
	const Finfo* targetFinfo = dest->findFinfo( targetFname );
	if ( targetFinfo == 0 ) {
		// Send back failure report
		return;
	}

	if ( targetFinfo->ftype()->typeStr() != typeSig ) {
		// Send back failure report
		return;
	}

	// Note that we could have used a different func here, but best
	// if we can keep it simple.
	
	const Finfo* parFinfo = self->findFinfo( "data" );
	assert( parFinfo != 0 );
	if ( !parFinfo->add( self, dest, targetFinfo /*, msgIndex */) ) {
		// Send back failure report
		return;
	}
	// send back success report.
}
#endif

/////////////////////////////////////////////////////////////////////
// This function does the main work of sending incoming messages
// to dests.
/////////////////////////////////////////////////////////////////////

/**
 * This function posts a non-blocking receive for data
 */
void PostMaster::innerPostIrecv()
{
	// cout << "!" << flush;
	// cout << "inner PostIrecv on node " << localNode_ << " from " << remoteNode_ << endl << flush;
	request_ = comm_->Irecv(
		&( recvBuf_[0] ), recvBuf_.size(), MPI_CHAR, 
			remoteNode_, DATA_TAG );
	// cout << inBufSize_ << " innerPostIrecv: request_ empty?" << ( request_ == static_cast< MPI::Request >( 0 ) ) << "\n";
	donePoll_ = 0;
}

void PostMaster::postIrecv( const Conn* c, int ordinal )
{
	static_cast< PostMaster* >( c->data() )->innerPostIrecv();
}


/**
 * Transmits data to proxy element, which then sends data out to targets.
 */
unsigned int proxy2tgt( const AsyncStruct& as, const char* data )
{
	// Make a Conn
	// call appropriate proxyFunc from the AsyncStruct, which could
	// be improved here.
	// Figure out how much data was sent and return that.
	return 0;
}

/**
 * This function does the main work of sending incoming messages
 * to dests. It grinds through the irecv'ed data and sends it out
 * to local dest objects.
 * This is called by the poll function
 */
void PostMaster::innerPoll( const Conn* c )
{
	Eref e = c->target();
	unsigned int pollMsgIndex = c->targetIndex();
	// Look up the irecv'ed data here
	// cout << "inner Poll on node " << localNode_ << " from " << remoteNode_ << endl << flush;
	if ( donePoll_ )
			return;
	if ( !request_ ) {
		sendTo1< unsigned int >( e, pollSlot, pollMsgIndex, remoteNode_ );
		donePoll_ = 1;
		return;
	}
	if ( request_.Test( status_ ) ) {
		// Data has arrived. How big was it?
		unsigned int dataSize = status_.Get_count( MPI_CHAR );
		// cout << dataSize << " bytes of data arrived on " << localNode_ << " from " << remoteNode_ << endl << flush;
		request_ = 0;
		if ( dataSize < sizeof( unsigned int ) ) return;

		// Handle async data in the buffer
		const char* data = &( recvBuf_[ 0 ] );
		unsigned int nMsgs = *static_cast< const unsigned int* >(
						static_cast< const void* >( data ) );
		data += sizeof( unsigned int );
		for ( unsigned int i = 0; i < nMsgs; i++ ) {
			AsyncStruct as( data );
			data += sizeof( AsyncStruct );
			unsigned int size = proxy2tgt( as, data );
			// assert( size == as.size() );
			data += size;
		}

		// Here we skip the location for the msgId, so this is only for
		// async msgs.
		data = &( recvBuf_[0] );
		const char* dataEnd = &( recvBuf_[ dataSize ] );
		while ( data < dataEnd ) {
			data++; // dummy
		}

		donePoll_ = 1;
	}
}

void PostMaster::poll( const Conn* c, int ordinal )
{
	static_cast< PostMaster* >( c->data() )->innerPoll( c );
}

/**
 * This uses a blocking send to send out the data. Normally this
 * should be immediate provided the iRecv has been posted at the
 * destination.
 */
void PostMaster::innerPostSend( )
{
	// send out the filled buffer here to the other nodes..
	// cout << "*" << flush;
	// cout << "sending " << outBufPos_ << " bytes: " << outBuf_ << endl << flush;
	comm_->Send( &( sendBuf_[0] ), sendBufPos_, 
		MPI_CHAR, remoteNode_, DATA_TAG
	);
	sendBufPos_ = 0;
}

void PostMaster::postSend( const Conn* c, int ordinal )
{
	static_cast< PostMaster* >( c->data() )->innerPostSend();
}

/////////////////////////////////////////////////////////////////////

#if 0
/**
 * This static function handles requests to set up a message. It does the
 * following:
 * - posts irecv for return status of the message
 * - Sends out the message request itself
 * - Creates the local message stuff
 * - Returns index for forming local message.
 */

unsigned int PostMaster::respondToAdd(
		Element* e, const string& respondString, unsigned int numDest )
{
		/*
		cout << "\nresponding to add from node " <<
			PostMaster::getMyNode( e ) <<
		 	" to node " << 
			PostMaster::getRemoteNode( e ) <<
			" with " << respondString << endl;
			*/
	// A little risky: We assume that the second message on the
	// dataSlot of the postmaster is the connection to the Shell,
	// and that nothing is inserted below this.
	unsigned int shellIndex = 0;
	PostMaster* p = static_cast< PostMaster* >( e->data() );
	char* buf = static_cast< char* >(
		p->innerGetAsyncParBuf( shellIndex, respondString.length() + 1 )
	);
	strcpy( buf, respondString.c_str() );

	return dataSlot;
}

/**
 * This function handles addition of message from postmaster to
 * targets. Usually called from within the PostMaster following a
 * cross-node message
 */
void PostMaster::placeIncomingFuncs( 
		vector< IncomingFunc >& inFl, unsigned int msgIndex )
{
	;
}
/////////////////////////////////////////////////////////////////////

#endif
void* getParBuf( const Conn* c, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c->data() );
	assert( pm != 0 );
//	return pm->innerGetParBuf( c.targetIndex(), size );
	return 0;
}

void* getAsyncParBuf( const Conn* c, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c->data() );
	assert( pm != 0 );
	return pm->innerGetAsyncParBuf( c, size );
}

/**
 * This function puts in the id of the message into the data buffer
 * and passes the next free location over to the calling function.
 * It internally increments the current location of the buffer.
 * If we don't use MPI, then this whole file is unlikely to be compiled.
 * So we define the dummy version of the function in DerivedFtype.cpp.
 */
void* PostMaster::innerGetAsyncParBuf( const Conn* c, unsigned int size )
{
	if ( size + sendBufPos_ > sendBuf_.size() ) {
		cout << "in getParBuf: Out of space in sendBuf_\n";
		// Do something clever here to send another installment
		return 0;
	}
	AsyncStruct as( c->target().id(), c->targetMsg(), c->sourceMsg() );
	char* sendBufPtr = &( sendBuf_[ sendBufPos_ ] );
	*static_cast< AsyncStruct* >( static_cast< void* >( sendBufPtr ) ) = as;
	sendBufPtr += sizeof( AsyncStruct );
	sendBufPos_ += sizeof( AsyncStruct ) + size;
	return static_cast< void* >( sendBufPtr );
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"
#include "Ftype2.h"
#include "setget.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"
#include "../shell/Shell.h"

extern void testMess( Element* e, unsigned int numNodes );

static Slot iSlot;
static Slot xSlot;
static Slot sSlot;
static Slot idVecSlot;
static Slot sVecSlot;

class TestParClass {
	public:
		static void sendI( const Conn* c ) {
			TestParClass* tp = static_cast< TestParClass* >( c->data() );
			send1< int >( c->target(), iSlot, tp->i_ ) ;
		}
		static void sendX( const Conn* c ) {
			TestParClass* tp = static_cast< TestParClass* >( c->data() );
			send1< double >( c->target(), xSlot, tp->x_ ) ;
		}
		static void sendS( const Conn* c ) {
			TestParClass* tp = static_cast< TestParClass* >( c->data() );
			send1< string >( c->target(), sSlot, tp->s_ ) ;
		}
		static void sendIdVec( const Conn* c ) {
			TestParClass* tp = static_cast< TestParClass* >( c->data() );
			send1< vector< Id > >( c->target(), idVecSlot, tp->idVec_ ) ;
		}
		static void sendSvec( const Conn* c ) {
			TestParClass* tp = static_cast< TestParClass* >( c->data() );
			send1< vector< string > >( c->target(), sVecSlot, tp->sVec_ ) ;
		}

		static void setI( const Conn* c, int value )
		{
			static_cast< TestParClass* >( c->data() )->i_ = value;
		}

		static void setX( const Conn* c, double value )
		{
			static_cast< TestParClass* >( c->data() )->x_ = value;
		}

		static void setS( const Conn* c, string value )
		{
			static_cast< TestParClass* >( c->data() )->s_ = value;
		}

		static void setIdVec( const Conn* c, vector< Id > value )
		{
			static_cast< TestParClass* >( c->data() )->idVec_ = value;
		}

		static void setSvec( const Conn* c, vector< string > value )
		{
			static_cast< TestParClass* >( c->data() )->sVec_ = value;
		}

		int i_;
		double x_;
		string s_;
		vector< Id > idVec_;
		vector< string > sVec_;
};

/**
 * This test class has messages for ints, doubles, strings, vectors.
 */
const Cinfo* initTestParClass()
{
	static Finfo* testParClassFinfos[] =
	{
		new DestFinfo( "i", Ftype1< int >::global(),
			RFCAST( &TestParClass::setI ) ),
		new DestFinfo( "x", Ftype1< double >::global(),
			RFCAST( &TestParClass::setX ) ),
		new DestFinfo( "s", Ftype1< string >::global(),
			RFCAST( &TestParClass::setS ) ),
		new DestFinfo( "idVec", Ftype1< vector< Id > >::global(),
			RFCAST( &TestParClass::setIdVec ) ),
		new DestFinfo( "sVec", Ftype1< vector< string > >::global(),
			RFCAST( &TestParClass::setSvec ) ),

		new SrcFinfo( "iSrc", Ftype1< int >::global() ),
		new SrcFinfo( "xSrc", Ftype1< double >::global() ),
		new SrcFinfo( "sSrc", Ftype1< string >::global() ),
		new SrcFinfo( "idVecSrc", Ftype1< vector< Id > >::global() ),
		new SrcFinfo( "sVecSrc", Ftype1< vector< string > >::global() ),
	};

	static Cinfo testParClassCinfo(
		"TestPar",
		"Upi Bhalla, 2008, NCBS",
		"Test class for parallel messaging",
		initNeutralCinfo(),
		testParClassFinfos,
		sizeof( testParClassFinfos ) / sizeof( Finfo* ),
		ValueFtype1< TestParClass >::global()
	);
	return & testParClassCinfo;
}

void testParAsyncMessaging()
{
	cout << "\nTesting Parallel Async messaging\n";
	const Cinfo* tpCinfo = initTestParClass();
	iSlot = tpCinfo->getSlot( "iSrc" );
	xSlot = tpCinfo->getSlot( "xSrc" );
	sSlot = tpCinfo->getSlot( "sSrc" );
	idVecSlot = tpCinfo->getSlot( "idVecSrc" );
	sVecSlot = tpCinfo->getSlot( "sVecSrc" );

	FuncVec::sortFuncVec();

	Element* n = Neutral::create( "Neutral", "n", Id(), Id::scratchId() );

	Element* p = Neutral::create(
			"PostMaster", "node0", n->id(), Id::scratchId());
	ASSERT( p != 0, "created Test Postmaster" );

	Element* t = 
		Neutral::create( "TestPar", "tp", n->id(), Id::scratchId() );
	ASSERT( t != 0, "created TestPar" );
	TestParClass* tdata = static_cast< TestParClass* >( t->data() );

	SetConn c( t, 0 );

	PostMaster* pm = static_cast< PostMaster* >( p->data() );

	// Send an int to the postmaster
	tdata->i_ = 44332211;
	bool ret = Eref( t ).add( "iSrc", p, "async" );
	ASSERT( ret, "msg to post" );
	TestParClass::sendI( &c );
	char* abuf = &( pm->sendBuf_[0] );
	AsyncStruct as( abuf );
	int asyncMsgNum = p->findFinfo( "async" )->msg();
	int iSendMsgNum = t->findFinfo( "iSrc" )->msg();
	ASSERT( as.target() == p->id(), "async info to post" );
	ASSERT( as.targetMsg() == asyncMsgNum, "async info to post" );
	ASSERT( as.sourceMsg() == iSendMsgNum, "async info to post" );

	void* vabuf = static_cast< void* >( abuf + sizeof( AsyncStruct ) );
	int i = *static_cast< int* >( vabuf );
	ASSERT( i == tdata->i_, "sending int" );

	// Send a double to the postmaster
	// Send a string to the postmaster
	// Send a vector of ints to the postmaster
	// Send a vector of Ids to the postmaster
	// Send a vector of strings to the postmaster

	// Make a message to an int via a proxy
	// Make a message to 100 ints via the proxy
	// Send a value to an int through the proxy
	//
	// Repeat for a vector of strings or something equally awful.
	//

	// Using a buffer-to-buffer memcpy, do a single-node check on the
	// complete transfer from int to int and all of the above.
	//
	// Also confirm that they work with multiple messages appearing
	// in random order each time.
	set( n, "destroy" );
}

void testPostMaster()
{
	// First, ensure that all nodes are synced.
	testParAsyncMessaging();
	MPI_INTRA_COMM.Barrier();
	unsigned int myNode = MPI_INTRA_COMM.Get_rank();
	unsigned int numNodes = MPI_INTRA_COMM.Get_size();
	Id* postId = new Id[numNodes];
	Eref post;
	unsigned int i;
	if ( myNode == 0 )
		cout << "\nTesting PostMaster: " << numNodes << " nodes";
	MPI_INTRA_COMM.Barrier();
	///////////////////////////////////////////////////////////////
	// check that we have postmasters for each of the other nodes
	// Print out a dot for each node.
	///////////////////////////////////////////////////////////////
	Id postMastersId =
			Neutral::getChildByName( Element::root(), "postmasters" );
	ASSERT( !postMastersId.bad(), "postmasters element creation" );
	Element* pms = postMastersId();
	for ( i = 0; i < numNodes; i++ ) {
		char name[10];
		sprintf( name, "node%d", i );
		Id id = Neutral::getChildByName( pms, name );
		if ( myNode == i ) { // Should not exist locally.
			ASSERT( id.bad(), "Checking local postmasters" )
		} else {
			ASSERT( !id.bad(), "Checking local postmasters" )
			Element* p = id();
			// cout << "name of what should be a postmaster: " << p->name() << endl << flush;
			ASSERT( p->className() == "PostMaster", "Check PostMaster");
			unsigned int remoteNode;
			get< unsigned int >( p, "remoteNode", remoteNode );
			ASSERT( remoteNode == i, "CheckPostMaster" );
		}
		postId[i] = id;
	}
	MPI_INTRA_COMM.Barrier();
	
	///////////////////////////////////////////////////////////////
	// This next test works on a single node too, for debugging.
	// In the single node case it fudges things to look like 2 nodes.
	// It tries to create a message from a table to a postmaster.
	///////////////////////////////////////////////////////////////
	// On all nodes, create a table and fill it up.
	
	Element* table = Neutral::create( "Table", "tab", Id(), 
		Id::scratchId() );
	// cout << myNode << ": tabId = " << table->id() << endl;
	ASSERT( table != 0, "Checking data flow" );
	set< int >( table, "xdivs", 10 );
	if ( myNode == 0 ) {
		set< int >( table, "stepmode", 2 ); // TAB_ONCE
		for ( unsigned int i = 0; i <= 10; i++ )
			lookupSet< double, unsigned int >( 
							table, "table", i * i, i );
	} else {
		set< int >( table, "stepmode", 3 ); // TAB_BUF
		for ( unsigned int i = 0; i <= 10; i++ )
			lookupSet< double, unsigned int >( 
							table, "table", 0.0, i );
	}

	MPI_INTRA_COMM.Barrier();

	if ( myNode == 0 ) {
		// Here we are being sneaky because we have the same id on all 
		// nodes.
		for ( i = 1; i < numNodes; i++ ) {
			post = postId[i].eref();
			set< Id >( post, "targetId", table->id() );
			set< string >( post, "targetField", "msgInput" );
			bool ret = Eref( table ).add( "outputSrc", post, "data" );
			ASSERT( ret, "Node 0 Making input message to postmaster" );
		}
	}

	// This section sends the data over to the remote node.
	Id cjId( "/sched/cj", "/" );
	assert( !cjId.bad() );
	Element* cj = cjId();
	set< double >( cj, "start", 1.0 );

	MPI_INTRA_COMM.Barrier();
	set( table, "destroy" );
	// unsigned int cjId = Shell::path2eid( "/sched/cj", "/" );

	////////////////////////////////////////////////////////////////
	// Now we fire up the scheduler on all nodes to keep info flowing.
	////////////////////////////////////////////////////////////////
	MPI_INTRA_COMM.Barrier();
	// sleep( 5 );
	MPI_INTRA_COMM.Barrier();
	char sendstr[50];

	for ( i = 0; i < numNodes; i++ ) {
		if ( i == myNode )
			continue;
		post = postId[i]();
		// PostMaster* pdata = static_cast< PostMaster* >( post->data() );
		sprintf( sendstr, "My name is Michael Caine %d,%d", myNode, i );

		// Find the Conn# of the message to the shell. Assume same
		// index is used on all nodes.
		// unsigned int shellIndex = 2;
//		cout << "dataslot = " << dataSlot << ", shellIndex = " << shellIndex << ", sendstr = " << sendstr << endl << flush;
		// char* buf = static_cast< char* >( pdata->innerGetAsyncParBuf( shellIndex, strlen( sendstr ) + 1 ));
		// strcpy( buf, sendstr );
	}
	MPI_INTRA_COMM.Barrier();
	bool glug = 0; // Breakpoint for parallel debugging
	while ( glug ) ;
	// cout << " starting string send\n" << flush;
	set< double >( cj, "start", 1.0 );
	// cout << " Done string send\n" << flush;
	MPI_INTRA_COMM.Barrier();

	////////////////////////////////////////////////////////////////
	// Now we set up a fully connected network of tables. Each 
	// node has numNodes tables. The one with index myNode is a source
	// the others are destinations. On each node the myNode table 
	// sends messages to tables with the same index on other nodes, 
	// to transfer its data to them. 
	//
	// We manually set up the msgs from
	// tables to postmaster and from postmaster to tables
	// This tests the full message flow process.
	// It also tests that info goes to multiple targets correctly.
	// It also tests multiple time-step transfer.
	//
	// We have to be careful that ordering of messages matches on
	// src and dest nodes. This works if we first create messages
	// with the lower table index, regardless of whether the
	// message is to or from the table.
	////////////////////////////////////////////////////////////////
	Element* n = Neutral::create( "Neutral", "n", Id(), Id::scratchId() );
	vector< Element* > tables( numNodes, 0 );
	Id tickId;
	lookupGet< Id, string >( cj, "lookupChild", tickId, "t0" );
	Element* tick = tickId();
	const Finfo* tickProcFinfo = tick->findFinfo( "outgoingProcess" );
	assert( tickProcFinfo != 0 );
	for ( i = 0; i < numNodes; i++ ) {
		char tabname[20];
		sprintf( tabname, "tab%d", i );
		tables[ i ] = Neutral::create( "Table", tabname, n->id(), Id::scratchId() );
		ASSERT( tables[i] != 0, "Checking data flow" );
		const Finfo* outFinfo = tables[i]->findFinfo( "outputSrc" );
		const Finfo* inFinfo = tables[i]->findFinfo( "input" );
		set< int >( tables[i], "xdivs", 10 );
		set< double >( tables[i], "xmin", 0.0 );
		set< double >( tables[i], "xmax", 10.0 );
		set< double >( tables[i], "input", 0.0 );
		bool ret = tickId.eref().add( "outgoingProcess", 
			tables[i], "process" );
		ASSERT( ret, "scheduling tables" );

		if ( i == myNode ) { // This is source table
			set< int >( tables[i], "stepmode", 2 ); // TAB_ONCE
			set< double >( tables[i], "stepsize", 1.0 ); // TAB_ONCE
			for ( unsigned int k = 0; k <= 10; k++ )
				lookupSet< double, unsigned int >( 
								tables[i], "table", i * 10 + k, k );

			for ( unsigned int j = 0; j < numNodes; j++ ) {
				if ( j == myNode ) continue;
				Element* p = postId[j]();
				// const Finfo* dataFinfo = p->findFinfo( "data" );
				set< Id >( p, "targetId", tables[i]->id() );
				set< string >( p, "targetField", "msgInput" );
				// bool ret = outFinfo->add( tables[i], p, dataFinfo );
				ret = Eref( tables[i] ).add( "outputSrc", 
					p, "data" );
				ASSERT( ret, "Making input message to postmaster" );
	MPI_INTRA_COMM.Barrier();
			}
		} else {
			post = postId[i]();
			const Finfo* dataFinfo = post->findFinfo( "data" );
			set< int >( tables[i], "stepmode", 3 ); // TAB_BUF
			set< double >( tables[i], "output", 0.0 ); // TAB_BUF
			for ( unsigned int k = 0; k <= 10; k++ )
				lookupSet< double, unsigned int >( 
								tables[i], "table", 0.0, k );

			ret = postId[i].eref().add( "data", tables[i], "input" );
			// bool ret = dataFinfo->add( post, tables[i], inFinfo );
			ASSERT( ret, "Making output message from postmaster" );
		}
	}
	set< double >( cj, "start", 11.0 );
	MPI_INTRA_COMM.Barrier();
	// At this point the contents of the tables should be changed by the
	// arrival of data.
	double value;
	for ( i = 0; i < numNodes; i++ ) {
		for ( unsigned int k = 0; k <= 10; k++ ) {
			lookupGet< double, unsigned int >( 
							tables[i], "table", value, k );
			// cout << "value = " << value << ", i = " << i << ", j = " << k << endl;
			if ( i == myNode ) {
				ASSERT( value == i * 10 + k , "Testing data transfer\n" );
			} else if ( k == 0 ) { // The value is delayed by one, and the first is zero
				ASSERT( value == 0 , "Testing data transfer\n" );
			} else {
				ASSERT( value == i * 10 + k - 1, "Testing data transfer\n");
			}
		}
	}
	set( n, "destroy" );
	MPI_INTRA_COMM.Barrier();

	Id shellId( "/shell", "/" );
	Element* shell = shellId();
	if ( myNode == 0 ) {
		testMess( shell, numNodes );
	}
}
#endif

#endif
