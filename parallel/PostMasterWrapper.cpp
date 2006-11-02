/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/*
struct PostBuffer {
	public: 
		unsigned int schedule;
		unsigned long size;
		vector< unsigned int > offset_;
	private:
		char* buffer;
};
*/

#define DATA_TAG 0
#define ASYNC_TAG 1
#define FUNC_TAG 2

// Dummy functions
// void isend( char* buf, int size, char* name, int tag, int dest );
// void irecv( char* buf, int size, char* name, int tag, int src );


#include "header.h"
#include "ParallelMsgSrc.h"
#include "ParallelFinfo.h"
#include "PostMaster.h"
#include "PostMasterWrapper.h"

// forward declaration
Element* traverseSrcToTick( Field& f );

Finfo* PostMasterWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"remoteNode", &PostMasterWrapper::getRemoteNode,
		&PostMasterWrapper::setRemoteNode, "int" ),
	new ReadOnlyValueFinfo< int >(
		"localNode", &PostMasterWrapper::getLocalNode, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new ParallelSrcFinfo(
		"srcOut", &PostMasterWrapper::getSrcSrc, 
		"" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new ParallelDestFinfo(
		"destIn", &PostMasterWrapper::destFunc,
		&PostMasterWrapper::getDestInConn, "" ),
	new Dest1Finfo< ProcInfo >(
		"postIn", &PostMasterWrapper::postFunc,
		&PostMasterWrapper::getPostConn, "", 1 ),
	new Dest0Finfo(
		"postInitIn", &PostMasterWrapper::postInitFunc,
		&PostMasterWrapper::getPostConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &PostMasterWrapper::processFunc,
		&PostMasterWrapper::getProcessConn, "", 1 ),
	new Dest0Finfo(
		"reinitIn", &PostMasterWrapper::reinitFunc,
		&PostMasterWrapper::getProcessConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"post", &PostMasterWrapper::getPostConn,
		"postIn, postInitIn" ),
	new SharedFinfo(
		"process", &PostMasterWrapper::getProcessConn,
		"processIn, reinitIn" ),
};

const Cinfo PostMasterWrapper::cinfo_(
	"PostMaster",
	"Upinder S. Bhalla, September 2006, NCBS",
	"PostMaster: Object for relaying messages between nodes. ",
	"Neutral",
	PostMasterWrapper::fieldArray_,
	sizeof(PostMasterWrapper::fieldArray_)/sizeof(Finfo *),
	&PostMasterWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// Can we have a single buffer and require that all entries into
// the buffer be complete within a tick? The idea is to have 
// the send go as soon as the entries finish.
// For recvs, we'll post the blocking recv at the last minute,
// but there may be better ways to do this.
void PostMasterWrapper::postFuncLocal( ProcInfo info )
{
	if ( outgoingEntireSize_.size() > tickCounter_ && 
					outgoingEntireSize_[ tickCounter_ ] > 0 ) {
		isend( &(outgoing_.front()),
			outgoingEntireSize_[ tickCounter_ ], "char", DATA_TAG, remoteNode_);
	}

	if ( incomingEntireSize_.size() > tickCounter_ && 
					incomingEntireSize_[ tickCounter_ ] > 0 ) {
		irecv( &(incoming_.front()),
			incomingEntireSize_[ tickCounter_ ], 
			"char", DATA_TAG, remoteNode_);

		srcSrc_.send( &( incoming_.front() ), incomingSchedule_, tickCounter_ );
	}
	++tickCounter_;
	if ( tickCounter_ >= numTicks_ )
			tickCounter_ = 0;
}

void PostMasterWrapper::processFuncLocal( ProcInfo info )
{
		// This is going to deal with the variable size messages
		// like action potls and shell commands.
	// If we have a clearly separate starter process tick for the
	// postmasters, this would be a good place to zero the tick Counter.
	// tickCounter_ = 0;
}

void PostMasterWrapper::reinitFuncLocal(  )
{
	checkPendingRequests();
	if ( needsReinit_ ) {
		vector< unsigned long > segments( 1, 0 );
		unsigned long nMsgs = outgoingSize_.size();
		unsigned int offset = 0;
		segments[0] = nMsgs;
		outgoingOffset_.resize( nMsgs );
		for (unsigned long i = 0; i < nMsgs; i++ ) {
			outgoingOffset_[i] = offset;
			offset += outgoingSize_[i];
		}
		outgoing_.resize( offset );
		//destInConn_.resize( segments );
		// processInConn_.resize( segments );
		assignSchedule(); 
		informTargetNode(); // Sends out schedule info.
		assignIncomingSchedule();
		needsReinit_ = 0;
	}
	tickCounter_ = 0;
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnPostMasterLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PostMasterWrapper, processConn_ );
	return reinterpret_cast< PostMasterWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
PostMasterWrapper::PostMasterWrapper(const string& n)
		:
			Neutral( n ),
			srcSrc_( &srcOutConn_ ),
			postConn_( this ),
			srcOutConn_( this ),
			destInConn_( this )
{
	vector< unsigned long > segments( 1, 0 );
	segments[0] = 4;
	destInConn_.resize( segments );
	// processInConn_.resize( segments );
	outgoingOffset_.resize( 1, 0 );
	outgoing_.resize( 40 );
	outgoingEntireSize_.resize( 1, 0 );
	incoming_.resize( 40 );
	incomingEntireSize_.resize( 1, 0 );
	needsReinit_ = 1;

	char temp[2];
	temp[0] = n[1];
	temp[1] = '\0';
	localNode_ = atoi( temp );
}
char* PostMasterWrapper::getPostPtr( unsigned long index )
{
	static char dummy[1024]; 
	if ( index < outgoingOffset_.size() )
		return &outgoing_[ outgoingOffset_[index] ];
	cerr << "Error: bad index " << index << 
	" for postmaster " << name() << "\n";
	return dummy;
}
char* getPostPtr( Conn* c )
{
	static char dummy[1024]; 
	PostMasterWrapper* pm = 
		dynamic_cast< PostMasterWrapper* >( c->parent() );
	if ( pm ) {
		// SolveMultiConn* mc = dynamic_cast< SolveMultiConn* >( c );
		SolverConn* sc = dynamic_cast< SolverConn* >( c );
		if ( sc )
			return pm->getPostPtr( sc->index() );
	}
	return dummy;
}

void PostMasterWrapper::isend( char* buf, int size, char* name, int tag, int dest )
{
	if ( tag == DATA_TAG )
		cout << "in isend data, with " << size << " bytes\n";
	else if ( tag == ASYNC_TAG )
		cout << "in isend addmsg to " << buf << " with " << size << " bytes\n";
	memcpy( remoteTransferBuffer_, buf, size );
}

void PostMasterWrapper::irecv( char* buf, int size, char* name, int tag, int src )
{
	cout << "in irecv, expecting " << size << " bytes\n";
	memcpy( buf, localTransferBuffer_, size );
}

void PostMaster::addSender( const Finfo* sender )
{
	outgoingSize_.push_back( sender->ftype()->size() );
	needsReinit_ = 1;
}

bool PostMasterWrapper::callsMe( Element* tickElm )
{
	Field me( this, "post" );
	Field tick( tickElm, "process" );
	vector< Field > list;
	me.dest( list );
	if ( list.size() == 0 )
		return 0;
	return ( std::find( list.begin(), list.end(), tick ) != list.end() );
}

/*
 *
 * #include mpi.h
 * bool MPI::Comm::Iprobe(int source, int tag) const;
 *
 * #include mpi.h
 * bool MPI::Comm::Iprobe(int source, int tag, MPI::Status& status) const;
 *
 *
 * #include mpi.h
 * bool MPI::Request::Test();
 *
 * #include mpi.h
 * bool MPI::Request::Test(MPI::Status& status);
 *
 *
*/

// Alternate implementation: Assume that the postmaster starts up
// with a message going to the local shell. Instead of having a
// separate channel for the addmsg request, the request is just
// issued through the regular message. The only point is that we
// need to handle these messages asynchronously. It would be
// valuable if there were other kinds of shell requests being passed,
// if it is just addmsg we could do it either way.
// This is pretty clear. We don't want the postmaster to have to
// know about what the shells are talking. So we need to put in
// async messaging here, not polling for addmsgs.
void PostMasterWrapper::checkPendingRequests()
{
	// A bit tricky. We don't know ahead of time what is coming.
	// So the best we can do is to probe if anything is pending,
	// and then immediately issue a recv. This is a message to
	// the shell, so we send on that info.
		/*
	while ( probeMessages( remoteNode_, ASYNC_TAG, status ) ) {
		// Use a blocking recv as we already have the message waiting.
		// The status info holds size etc.
		irecv( buf, status.size(), "char", ASYNC_TAG, remoteNode_ );
	}
	*/
}

void PostMasterWrapper::connectTick( Element* tick )
{
	string tickName = "post_" + tick->name();
	string tickPath = "../post_" + tick->name();
	Element* src = tick->relativeFind( tickPath );
	double dt;
	double stage;
	if ( !src ) {
		// Make a new tick and connect it up.
		src = Cinfo::find("ClockTick")->create(
						tickName, tick->parent() );
		Ftype1<double>::get( tick, "dt", dt );
		Ftype1<double>::set( src, "dt", dt );
		Ftype1<double>::get( tick, "stage", stage );
		Ftype1<double>::set( src, "stage", stage + 0.5 );
		Element* job = tick->parent();
		Ftype0::set( job, job->field( "reschedIn" ).getFinfo() );
	}
	if ( !callsMe( src )  ) {
		Field proc( src, "process" );
		Field myproc( this, "post" );
		proc.add( myproc );
	}
}

void PostMasterWrapper::assignSchedule()
{
	vector< Field > srcList;
	// vector< Field >::iterator i;
	Field di = field( "destIn" );
	di.src( srcList );
	unsigned long i;
	vector< Element* > ticks;
	for ( i = 0; i < srcList.size(); i++ ) {
		ticks.push_back( traverseSrcToTick( srcList[i] ) );
		if ( ticks[i] )
			cout << srcList[i].name() << "	" << ticks[i]->name() << endl;
		else
			cout << srcList[i].name() << ": No source tick found\n";
	}
	vector< Element* > temp = ticks;
	vector< Element* > t2;
	sort( temp.begin(), temp.end() );
	unique_copy( temp.begin(), temp.end(), back_inserter( t2 ) );
	// Now I have all the ticks. Build up the schedule
	map< const Element*, int > schedule;
	vector< Element* >::iterator endseq = 
		unique( temp.begin(), temp.end() );
	vector< Element* >::iterator j;
	i = 0;
	for ( j = temp.begin(); j != endseq; ++j ) {
		schedule[ *j ] = i++;
		connectTick( *j );
	}

	numTicks_ = i;	

	outgoingSchedule_.resize( outgoingSize_.size() );
	for ( i = 0; i < ticks.size(); i++ ) {
		outgoingSchedule_[i] = schedule[ ticks[i] ];
	}

	// With the schedule in place I can now assign offsets. I will
	// reuse the buffer on each tick to save memory.
	long max_offset = 0;
	unsigned long k;
	outgoingEntireSize_.resize( numTicks_ );
	for ( i = 0; i < numTicks_; i++ ) {
		long offset = 0;
		for ( k = 0; k < outgoingSize_.size() ; k++ ) {
			if ( outgoingSchedule_[k] == i ) {
				outgoingOffset_[k] = offset;
				offset += outgoingSize_[k];
			}
		}
		outgoingEntireSize_[i] = offset;
		if ( offset > max_offset )
			max_offset = offset;
	}
	outgoing_.resize( max_offset );
}

// Issue here: Do we send stuff all at once, or as messages are
// defined?
// Let's set up messages on the go, far easier to debug. But we
// definitely need this stage to pass over the scheduling info.
void PostMasterWrapper::informTargetNode()
{
	// Let's be paranoid here. We first tell the target node
	// how many ticks, how many messages, and buffer size: 3 ints.
	// Then we send a message with the outgoingEntireSize and
	// outgoingSchedule vectors so we can double check all assignments.
	
	int info[3];
	info[0] = outgoingEntireSize_.size();
	info[1] = outgoingSchedule_.size();
	info[2] = outgoing_.size();
	char* temp = reinterpret_cast< char * >( info );
	isend( temp, 3 * sizeof( int ), "int", DATA_TAG, remoteNode_);
	// Here we receive the info from above. Has to be blocking.
	assignIncomingSizes();

	//
	// Target node needs to know:
	// Each object to connect to
	// 		Identity of object gives 
	// When to place request for data
	// 
	// Target node needs to figure out:
	// When to place request for data
	vector< unsigned int > data;
	vector< unsigned int >::iterator i;
	data = outgoingEntireSize_;
	for (i = outgoingSchedule_.begin();
					i != outgoingSchedule_.end(); i++)
		data.push_back( *i );


	temp = reinterpret_cast< char * >( &( data.front() ) );
	isend( temp,
		data.size() * sizeof( int ), "int", DATA_TAG, remoteNode_);
	// Here is the receipt function for the above. Again, blocking.
	assignIncomingSchedule();
}

// Here we should do a lot of cross checking with local info about
// sizes and numbers.
void PostMasterWrapper::assignIncomingSizes()
{
	int info[3];
	char* temp = reinterpret_cast< char * >( info );
	irecv( temp, 3 * sizeof( int ), "int", DATA_TAG, remoteNode_);

	cout << "orig data on node " << localNode_ << 
		": NumTicks=" << numTicks_ <<
		", numTgts=" << srcSrc_.nTargets() << 
		", BufSize=" << info[2] << endl;

	cout << "Received input from " << remoteNode_ << 
		": numTicks=" << info[0] <<
		", numTgts=" << info[1] << 
		", bufSize=" << info[2] << endl;


	if ( numTicks_ != static_cast< unsigned int >( info[0] ) )
		cout << "Error: numTicks != arriving numTicks on node " <<
				localNode_ << endl;

	if ( srcSrc_.nTargets() != static_cast< unsigned int >( info[1] ) )
		cout << "Error: nTargets != arriving nTargets on node " <<
				localNode_ << endl;

	// Here we do a hack to get the system to work without MPI.
	if ( localNode_ == 1 ) {
		info[0] = 0;
		info[1] = 0;
		info[2] = 0;
	} else {
		info[0] = 2;
		info[1] = 4;
		info[2] = 24;
		numTicks_ = info[0];	
	}
	
	incomingEntireSize_.resize( info[0] );
	incomingSchedule_.resize( info[1] );
	incoming_.resize( info[2] );
}

// This function receives the scheduling info from the remote node.
// We have just received the sizing info.
void PostMasterWrapper::assignIncomingSchedule()
{
	size_t totSize = incomingSchedule_.size() + incomingEntireSize_.size();
	vector< unsigned int >temp( totSize );
	char* ctemp = reinterpret_cast< char * >( &( temp.front() ) );

	irecv( ctemp, totSize * sizeof( int ),
					"int", DATA_TAG, remoteNode_);

	size_t i;
	for ( i = 0; i < incomingEntireSize_.size(); i++ )
		incomingEntireSize_[i] = temp[i];

	vector< unsigned int >::iterator j = incomingSchedule_.begin();
	for ( ; i < totSize; i++ )
		*j++ = temp[i];

	temp = incomingSchedule_;
	sort( temp.begin(), temp.end() );
	vector< unsigned int >::iterator lastunique =
		unique( temp.begin(), temp.end() );

	char tickname[200];
	for ( j = temp.begin(); j != lastunique; j++ ) {
		sprintf( tickname, "/sched/cj/ct%d", *j );
		Element* e = Element::find( tickname );
		connectTick( e );
		// There is a problem here if the postmaster both sends
		// and receives info. It is possible that the already set up
		// list of ticks will be changed by this function.
		// For now, leave it. We will later have a more systematic
		// way to connect up ticks.
	}
}

Element* traverseSrcToTick( Field& f )
{
	static const Cinfo* tickCinfo = Cinfo::find( "ClockTick" );
	if ( f.getElement()->cinfo() == tickCinfo )
		return f.getElement();

	vector< Field > srcList;
	vector< Field >::iterator i;
	f.src( srcList );
	Element* ret;
	for ( i = srcList.begin(); i != srcList.end(); i++ ) {
		ret = traverseSrcToTick( *i );
		if ( ret )
				return ret;
	}

	return 0;
}

// As a testing hack, we'll look up pointers to a data transfer
// buffer in the remote node so that we can dump data through isends.
void PostMasterWrapper::localSetRemoteNode( int node )
{
	char remoteName[10];
	sprintf( remoteName, "/p%d", node );
	Element* remote = Element::find( remoteName );
	PostMasterWrapper* pr = static_cast< PostMasterWrapper* >( remote );
	
	remoteTransferBuffer_ = pr->localTransferBuffer_;
	remoteNode_ = node;
}
