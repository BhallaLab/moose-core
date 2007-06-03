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
#include "ParFinfo.h"
#include <sstream>
#include <typeinfo>
#include "../element/Neutral.h" // Possibly only need this for testing.
#include "../shell/Shell.h" // Possibly only need this for testing.

#define DATA_TAG 0

/**
 * Declaration of the neutralCinfo() function is here because
 * we ensure the correct sequence of static initialization by having
 * each Cinfo use this call to find its base class. Most Cinfos
 * inherit from neutralCinfo. This function
 * uses the common trick of having an internal static value which
 * is created the first time the function is called.
 * The function for neutralCinfo has an additional line to statically
 * initialize the root element.
 */
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
		new DestFinfo( "postIrecv",
			Ftype1< int >::global(), RFCAST( &PostMaster::postIrecv ) ),
		// The second entry is to tell the PostMaster to post 'send'
		new DestFinfo( "postSend",
			Ftype1< int >::global(), RFCAST( &PostMaster::postSend ) ),
		// The third entry is for polling the receipt of incoming data.
		// Each PostMaster does an MPI_Test on the earlier posted iRecv.
		new DestFinfo( "poll",
			Ftype1< int >::global(), RFCAST( &PostMaster::poll ) ),
		// The fourth entry is for harvesting the poll request.
		// The argument is the node number handled by the postmaster.
		// It comes back when the polling on that postmaster is done.
		new SrcFinfo( "harvestPoll", Ftype1< unsigned int >::global() )
	};

	/*
	static TypeFuncPair parTypes[] = 
	{
		// This first entry is to tell the PostMaster to post iRecvs
		// The argument is the ordinal number of the clock tick
		TypeFuncPair( Ftype1< int >::global(), 
						RFCAST( &PostMaster::postIrecv ) ),
		// The second entry is to tell the PostMaster to post 'send'
		TypeFuncPair( Ftype1< int >::global(),
						RFCAST( &PostMaster::postSend ) ),
		// The third entry is for polling the receipt of incoming data.
		// Each PostMaster does an MPI_Test on the earlier posted iRecv.
		TypeFuncPair( Ftype1< int >::global(),
						RFCAST( &PostMaster::poll ) ),
		// The fourth entry is for harvesting the poll request.
		// The argument is the node number handled by the postmaster.
		// It comes back when the polling on that postmaster is done.
		TypeFuncPair( Ftype1< unsigned int >::global(), 0 )
	};
	*/

	static Finfo* serialShared[] =
	{
		new SrcFinfo( "rawAdd", // addmsg using serialized data.
			Ftype1< string >::global() ),
		new SrcFinfo( "rawCopy", // copy using serialized data.
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
		new ValueFinfo( "targetId", 
					ValueFtype1< unsigned int >::global(),
					GFCAST( &PostMaster::getTargetId ),
					RFCAST( &PostMaster::setTargetId )
		),
		new ValueFinfo( "targetField", 
					ValueFtype1< string >::global(),
					GFCAST( &PostMaster::getTargetField ),
					RFCAST( &PostMaster::setTargetField )
		),
		////////////////////////////////////////////////////////////////
		//	Special Finfo for the postmaster for handling serialization.
		////////////////////////////////////////////////////////////////
		new ParFinfo( "data" ),

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

static const unsigned int pollSlot = 
	initPostMasterCinfo()->getSlotIndex( "harvestPoll" );
static const unsigned int addSlot = 
	initPostMasterCinfo()->getSlotIndex( "rawAdd" );
static const unsigned int copySlot = 
	initPostMasterCinfo()->getSlotIndex( "rawCopy" );

//////////////////////////////////////////////////////////////////
// Here we put the PostMaster class functions.
//////////////////////////////////////////////////////////////////
PostMaster::PostMaster()
	: remoteNode_( 0 ), donePoll_( 0 ), comm_( &MPI::COMM_WORLD )
{
	localNode_ = MPI::COMM_WORLD.Get_rank();
	outBufSize_ = 10000;
	outBuf_ = new char[ outBufSize_ ];
	inBufSize_ = 10000;
	inBuf_ = new char[ inBufSize_ ];
	if ( incomingFunc_.size() == 0 ) {
		incomingFunc_.push_back( lookupFunctionData( RFCAST( Neutral::childFunc ) )->index() );
		incomingFunc_.push_back( lookupFunctionData( RFCAST( Shell::rawAddFunc ) )->index() );
		cout << "incoming func[0] = " << incomingFunc_[0] << endl;
	}
}

//////////////////////////////////////////////////////////////////
// Here we put the PostMaster Moose functions.
//////////////////////////////////////////////////////////////////

unsigned int PostMaster::getMyNode( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->localNode_;
}

unsigned int PostMaster::getRemoteNode( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->remoteNode_;
}

void PostMaster::setRemoteNode( const Conn& c, unsigned int node )
{
		static_cast< PostMaster* >( c.data() )->remoteNode_ = node;
}


unsigned int PostMaster::getTargetId( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->targetId_;
}

void PostMaster::setTargetId( const Conn& c, unsigned int value )
{
		static_cast< PostMaster* >( c.data() )->targetId_ = value;
}


string PostMaster::getTargetField( const Element* e )
{
		return static_cast< PostMaster* >( e->data() )->targetField_;
}

void PostMaster::setTargetField( const Conn& c, string value )
{
		static_cast< PostMaster* >( c.data() )->targetField_ = value;
}


/////////////////////////////////////////////////////////////////////
// Here we handle passing messages to off-nodes
/////////////////////////////////////////////////////////////////////

// Just to help me remember how to use the typeid from RTTI.
// This will work only between identical compilers, I think.
// Something more exciting happens with shared finfos.
const char* ftype2str( const Ftype *f )
{
	return typeid( *f ).name();
}


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

/**
 * This function puts in the id of the message into the data buffer
 * and passes the next free location over to the calling function.
 * It internally increments the current location of the buffer.
 * If we don't use MPI, then this whole file is unlikely to be compiled.
 * So we define the dummy version of the function in DerivedFtype.cpp.
 */
void* PostMaster::innerGetAsyncParBuf( 
				unsigned int targetIndex, unsigned int size )
{
	if ( size + outBufPos_ > outBufSize_ ) {
		cout << "in getParBuf: Out of space in outBuf.\n";
		// Do something clever here to send another installment
		return 0;
	}
	*static_cast< unsigned int* >( 
			static_cast< void* >( outBuf_ + outBufPos_ ) ) =
			targetIndex;
	outBufPos_ += sizeof( unsigned int ) + size;
	return static_cast< void* >( outBuf_ + outBufPos_ - size );
}

void PostMaster::parseMsgRequest( const char* req, Element* self )
{
	// sscanf( req, "%d %d %s", srcId, destId, typeSig );
	// string sreq( req );

	unsigned int srcId;
	unsigned int destId;
	string typeSig;
	string targetFname;
	unsigned int msgIndex;

	istringstream istr;
	istr.str( req );
	assert ( (istr >> srcId >> destId >> typeSig >> targetFname 
						>> msgIndex >> ws ) && istr.eof());
	assert ( srcId != BAD_ID );
	assert ( destId != BAD_ID );

	Element* dest = Element::element( destId );
	assert( dest != 0 );
	const Finfo* targetFinfo = dest->findFinfo( targetFname );
	if ( targetFinfo == 0 ) {
		// Send back failure report
		return;
	}

	if ( ftype2str( targetFinfo->ftype() ) != typeSig ) {
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

/////////////////////////////////////////////////////////////////////
// This function does the main work of sending incoming messages
// to dests.
/////////////////////////////////////////////////////////////////////

/**
 * This function posts a non-blocking receive for data
 */
void PostMaster::innerPostIrecv()
{
	cout << "!" << flush;
	request_ = comm_->Irecv(
			inBuf_, inBufSize_, MPI_CHAR, remoteNode_, DATA_TAG );
	donePoll_ = 0;
}

void PostMaster::postIrecv( const Conn& c, int ordinal )
{
	static_cast< PostMaster* >( c.data() )->innerPostIrecv();
}

/**
 * This function does the main work of sending incoming messages
 * to dests. It grinds through the irecv'ed data and sends it out
 * to local dest objects.
 * This is called by the poll function
 */
void PostMaster::innerPoll( Element* e)
{
	// Look up the irecv'ed data here
	// cout << "inner Poll\n" << flush;
	if ( donePoll_ )
			return;
	if ( !request_ ) {
		send1< unsigned int >( e, pollSlot, remoteNode_ );
		donePoll_ = 1;
		return;
	}
	if ( request_.Test( status_ ) ) {
		// Data has arrived. How big was it?
		unsigned int dataSize = status_.Get_count( MPI_CHAR );
		cout << dataSize << " bytes of data arrived on " << 
			localNode_ << " from " << remoteNode_ << endl << flush;
		request_ = 0;
		if ( dataSize < sizeof( unsigned int ) ) return;

		// Handle async data in the buffer
		/*
		unsigned int nMsgs = *static_cast< unsigned int *>(
						static_cast< void* >( inBuf_ ) );
		const char* data = inBuf_ + sizeof( unsigned int );
		for ( unsigned int i = 0; i < nMsgs; i++ ) {
		}
		*/
		// Here we skip the location for the msgId, so this is only for
		// async msgs.
		const char* data = inBuf_;
		while ( data < inBuf_ + dataSize ) {
			cout << "1:"<<localNode_ << "," << remoteNode_ << endl << flush;
			unsigned int msgId =  *static_cast< const unsigned int *>(
				static_cast< const void* >( data ) );
				// the funcVec_ has msgId entries matching each Conn
			data += sizeof( unsigned int );
			cout << "1.5:"<<localNode_ << "," << remoteNode_ <<
				"msgid = " << msgId << endl << flush;
			// Hack for testing: sometimes msgId comes in out of range.
			if ( msgId > incomingFunc_.size() )
				break;
			unsigned int funcId = incomingFunc_[ msgId ]; 
			RecvFunc rf = lookupFunctionData( funcId )->func();
			cout << "2:"<<localNode_ << "," << remoteNode_ << endl << flush;
			IncomingFunc pf = 
				lookupFunctionData( funcId )->funcType()->inFunc();
			cout << "3:"<<localNode_ << "," << remoteNode_ << endl << flush;
			data = static_cast< const char* >(
				pf( *( e->lookupConn( msgId ) ), data, rf ) );
			cout << "4:"<<localNode_ << "," << remoteNode_ << endl << flush;
			assert (data != 0 );
		}

		send1< unsigned int >( e, pollSlot, remoteNode_ );
		donePoll_ = 1;
	}
}

void PostMaster::poll( const Conn& c, int ordinal )
{
	static_cast< PostMaster* >( c.data() )->
			innerPoll( c.targetElement() );
}

/**
 * This uses a blocking send to send out the data. Normally this
 * should be immediate provided the iRecv has been posted at the
 * destination.
 */
void PostMaster::innerPostSend( )
{
	// send out the filled buffer here to the other nodes..
	cout << "*" << flush;
	cout << "sending " << outBufPos_ << " bytes: " << 
		outBuf_ << endl << flush;
	comm_->Send(
			outBuf_, outBufPos_, MPI_CHAR, remoteNode_, DATA_TAG
	);
	outBufPos_ = 0;
}

void PostMaster::postSend( const Conn& c, int ordinal )
{
	static_cast< PostMaster* >( c.data() )->innerPostSend();
}

/////////////////////////////////////////////////////////////////////
/**
 * This function handles requests to set up a message. It does the
 * following:
 * - posts irecv for return status of the message
 * - Sends out the message request itself
 * - Creates the local message stuff
 * - Returns index for forming local message.
 */

unsigned int PostMaster::respondToAdd(
		Element* e, const string& respondString, unsigned int numDest )
{
		cout << "\nresponding to add from node " <<
			PostMaster::getMyNode( e ) <<
		 	" to node " << 
			PostMaster::getRemoteNode( e ) <<
			" with " << respondString << endl;
	return 0;
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

void* getParBuf( const Conn& c, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c.data() );
	assert( pm != 0 );
	return pm->innerGetParBuf( c.targetIndex(), size );
}

void* getAsyncParBuf( const Conn& c, unsigned int size )
{
	PostMaster* pm = static_cast< PostMaster* >( c.data() );
	assert( pm != 0 );
	return pm->innerGetAsyncParBuf( c.targetIndex(), size );
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"
#include "Ftype2.h"
#include "setget.h"
#include "../builtins/Interpol.h"
#include "../builtins/Table.h"
#include "../shell/Shell.h"

void testPostMaster()
{
	// First, ensure that all nodes are synced.
	MPI::COMM_WORLD.Barrier();
	unsigned int myNode = MPI::COMM_WORLD.Get_rank();
	unsigned int numNodes = MPI::COMM_WORLD.Get_size();
	unsigned int* postId = new unsigned int[numNodes];
	if ( myNode == 0 )
		cout << "\nTesting PostMaster: " << numNodes << " nodes";
	MPI::COMM_WORLD.Barrier();
	///////////////////////////////////////////////////////////////
	// check that we have postmasters for each of the other nodes
	// Print out a dot for each node.
	///////////////////////////////////////////////////////////////
	unsigned int postMastersId =
			Neutral::getChildByName( Element::root(), "postmasters" );
	ASSERT( postMastersId != BAD_ID, "postmasters element creation" );
	Element* pms = Element::element( postMastersId );
	for ( unsigned int i = 0; i < numNodes; i++ ) {
		char name[10];
		sprintf( name, "node%d", i );
		unsigned int id = Neutral::getChildByName( pms, name );
		if ( myNode == i ) { // Should not exist locally.
			ASSERT( id == BAD_ID, "Checking local postmasters" )
		} else {
			ASSERT( id != BAD_ID, "Checking local postmasters" )
			Element* p = Element::element( id );
			ASSERT( p->className() == "PostMaster", "Check PostMaster");
			unsigned int remoteNode;
			get< unsigned int >( p, "remoteNode", remoteNode );
			ASSERT( remoteNode == i, "CheckPostMaster" );
		}
		postId[i] = id;
	}
	MPI::COMM_WORLD.Barrier();
	
	///////////////////////////////////////////////////////////////
	// This next test works on a single node too, for debugging.
	// In the single node case it fudges things to look like 2 nodes.
	// It tries to create a message from a table to a postmaster.
	///////////////////////////////////////////////////////////////
	// On all nodes, create a table and fill it up.
	Element* table = Neutral::create( "Table", "tab", Element::root() );
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

	Element* post;
	unsigned int i;
	if ( numNodes == 1 ) { // Create a dummy postmaster
		post = Neutral::create( "PostMaster", "node1", pms );
		numNodes = 2;
		delete[] postId;
		postId = new unsigned int[numNodes];
		postId[0] = BAD_ID;
		postId[1] = post->id();
	}
	if ( myNode == 0 ) {
		// Here we are being sneaky because we have the same id on all 
		// nodes.
		for ( i = 1; i < numNodes; i++ ) {
			post = Element::element( postId[i] );
			set< unsigned int >( post, "targetId", table->id() );
			set< string >( post, "targetField", "input" );
			const Finfo* outFinfo = table->findFinfo( "outputSrc" );
			const Finfo* dataFinfo = post->findFinfo( "data" );
			bool ret = outFinfo->add( table, post, dataFinfo );
			ASSERT( ret, "Node 0 Making input message to postmaster" );
		}
	}


	// This first test sends data from node 0 to all the other
	// nodes using hard-coded messaging. We use a 
	// table at each end for the source and dest of the data.
		/*
		cout << "\n ftype2str( Ftype1< double > ) = " <<
				ftype2str( Ftype1< double >::global() );
		cout << "\n ftype2str( Ftype2< string, vector< unsigned int > > ) = " <<
				ftype2str( Ftype2< string, vector< unsigned int > >::global() );
		cout << "\n ftype2str( ValueFtype1< Table >::global() ) = " <<
				ftype2str( ValueFtype1< Table >::global() );
				*/
	set( table, "destroy" );

	////////////////////////////////////////////////////////////////
	// Now we fire up the scheduler on all nodes to keep info flowing.
	////////////////////////////////////////////////////////////////
	MPI::COMM_WORLD.Barrier();
	unsigned int cjId = Shell::path2eid( "/sched/cj", "/" );
	assert( cjId != BAD_ID );
	Element* cj = Element::element( cjId );
	set< double >( cj, "start", 1.0 );
	char sendstr[50];
	for ( i = 0; i < numNodes; i++ ) {
		if ( i == myNode )
			continue;
		post = Element::element( postId[i] );
		PostMaster* pdata = static_cast< PostMaster* >( post->data() );
		sprintf( sendstr, "My name is Michael Caine %d,%d", myNode, i );

		// Find the Conn# of the message to the shell. Assume same
		// index is used on all nodes.
		unsigned int shellIndex = post->connSrcBegin( addSlot ) - 
			post->lookupConn( 0 );
		cout << "shellIndex = " << shellIndex << ", sendstr = " << sendstr << endl << flush;
		char* buf = static_cast< char* >(
			pdata->innerGetAsyncParBuf( shellIndex, strlen( sendstr ) + 1 )
		);
		strcpy( buf, sendstr );
	}
	set< double >( cj, "start", 1.0 );
}
#endif

#endif
