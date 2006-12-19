/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <mpi.h>
#include "ParallelMsgSrc.h"
#include "ParallelFinfo.h"
#include "PostMaster.h"
#include "PostMasterWrapper.h"


Finfo* PostMasterWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ReadOnlyValueFinfo< int >(
		"myNode", &PostMasterWrapper::getMyNode, "int" ),
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"pollFlag", &PostMasterWrapper::getPollFlag, 
		&PostMasterWrapper::setPollFlag, "int" ),
	new ValueFinfo< int >(
		"remoteNode", &PostMasterWrapper::getRemoteNode, 
		&PostMasterWrapper::setRemoteNode, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new ParallelSrcFinfo(
		"srcOut", &PostMasterWrapper::getSrcSrc, 
		"" ),
	new SingleSrc1Finfo< string >(
		"remoteCommandOut", &PostMasterWrapper::getRemoteCommandSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"pollRecvOut", &PostMasterWrapper::getPollRecvSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new ParallelDestFinfo(
		"destIn", &PostMasterWrapper::destFunc,
		&PostMasterWrapper::getDestInConn, "" ),
	new Dest1Finfo< int >(
		"ordinalIn", &PostMasterWrapper::ordinalFunc,
		&PostMasterWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< int >(
		"asyncIn", &PostMasterWrapper::asyncFunc,
		&PostMasterWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< int >(
		"postIrecvIn", &PostMasterWrapper::postIrecvFunc,
		&PostMasterWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< int >(
		"postSendIn", &PostMasterWrapper::postSendFunc,
		&PostMasterWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< int >(
		"pollRecvIn", &PostMasterWrapper::pollRecvFunc,
		&PostMasterWrapper::getParProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &PostMasterWrapper::processFunc,
		&PostMasterWrapper::getProcessConn, "", 1 ),
	new Dest0Finfo(
		"reinitIn", &PostMasterWrapper::reinitFunc,
		&PostMasterWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< string >(
		"remoteCommandIn", &PostMasterWrapper::remoteCommandFunc,
		&PostMasterWrapper::getRemoteCommandConn, "", 1 ),
	new Dest3Finfo< Field, int, int >(
		"addOutgoingIn", &PostMasterWrapper::addOutgoingFunc,
		&PostMasterWrapper::getRemoteCommandConn, "", 1 ),
	new Dest3Finfo< Field, int, int >(
		"addIncomingIn", &PostMasterWrapper::addIncomingFunc,
		&PostMasterWrapper::getRemoteCommandConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &PostMasterWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"remoteCommand", &PostMasterWrapper::getRemoteCommandConn,
		"remoteCommandIn, remoteCommandOut, addOutgoingIn, addIncomingIn" ),
	new SharedFinfo(
		"parProcess", &PostMasterWrapper::getParProcessConn,
		"ordinalIn, asyncIn, postIrecvIn, postSendIn, pollRecvIn, pollRecvOut" ),
};

const Cinfo PostMasterWrapper::cinfo_(
	"PostMaster",
	"Upinder S. Bhalla, November 2006, NCBS",
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
// EvalField function definitions
///////////////////////////////////////////////////

int PostMasterWrapper::localGetPollFlag() const
{
			return pollFlag_;
}
void PostMasterWrapper::localSetPollFlag( int value ) {
			pollFlag_ = value;
			while ( pollFlag_ )
				checkPendingRequests();
}
int PostMasterWrapper::localGetRemoteNode() const
{
			return remoteNode_;
}
void PostMasterWrapper::localSetRemoteNode( int value ) {
			remoteNode_ = value;
			asyncRequest_ = comm_.Irecv( 
				asyncBuf_, ASYNC_BUF_SIZE, MPI_CHAR,
				remoteNode_, ASYNC_TAG );
			// cout << "Making initial asyncReq on node " << myNode_ << endl;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void PostMasterWrapper::ordinalFuncLocal( int tick )
{
		// cout << "ordinal func on node " << myNode_ << 
		// 		" for tick " << tick << endl;
		unsigned long temp = tick + 1;
		if ( temp > outgoingEntireSize_.size() ) {
			outgoingEntireSize_.resize( temp, 0 );
			incomingEntireSize_.resize( temp, 0 );
		}
}

void PostMasterWrapper::postIrecvFuncLocal( int tick )
{
			unsigned long i = static_cast< unsigned long >( tick );
			if ( incomingEntireSize_.size() > i && 
				incomingEntireSize_[ tick ] > 0 ) { 
		//		cout << "Posting Irecv on node " << myNode_ << 
		//				" for node " << remoteNode_ << endl;
			request_ = comm_.Irecv(
				&(incoming_.front()),
				incomingEntireSize_[ tick ], 
				MPI_CHAR, remoteNode_, DATA_TAG );
			}
}
void PostMasterWrapper::postSendFuncLocal( int tick )
{
			unsigned long i = static_cast< unsigned long >( tick );
			if ( outgoingEntireSize_.size() > i && 
				outgoingEntireSize_[ tick ] > 0 ) { 
		//		cout << "Posting Send on node " << myNode_ << 
		//				" for node " << remoteNode_ << 
		//				", size= " << outgoingEntireSize_[tick] << endl;
				comm_.Send(
					&(outgoing_.front()),
					outgoingEntireSize_[ tick ], 
					MPI_CHAR,
					remoteNode_,
					DATA_TAG
				);
			}
}
void PostMasterWrapper::pollRecvFuncLocal( int tick )
{
	//		cout << "p";
			if ( !request_ || incomingEntireSize_[ tick ] == 0 ) {
				pollRecvSrc_.sendTo( tick );
				return;
			}
			if ( request_.Test() ) {	
	//			cout << "\nFinished polling Recv on node " << myNode_ << 
	//					" for node " << remoteNode_ << endl;
				srcSrc_.send( &( incoming_.front() ), incomingSchedule_, tick );
				pollRecvSrc_.sendTo( tick );
				request_ = 0;
			}
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
				// informTargetNode(); 
				// assignIncomingSchedule();
				needsReinit_ = 0;
			}
}
void PostMasterWrapper::remoteCommandFuncLocal( string data )
{
			checkPendingRequests();
			comm_.Send(
				data.c_str(),
				data.length(),
				MPI_CHAR,
				remoteNode_,
				ASYNC_TAG
			);
}

void PostMasterWrapper::countTicks()
{
	unsigned long nTicks = parProcessConn_.nTargets();
	if ( nTicks > outgoingEntireSize_.size() )
		outgoingEntireSize_.resize( nTicks );
	if ( nTicks > incomingEntireSize_.size() )
		incomingEntireSize_.resize( nTicks );
}

void PostMasterWrapper::addOutgoingFuncLocal( Field src, int tick, int size )
{
//			cout << "Postmaster on node " << myNode_ << 
//					" got addOutgoingFuncLocal( " << src.path() <<
//					", " << tick << ", " << size << ")\n";
		countTicks();
			unsigned int currSize = outgoingEntireSize_[tick];
			outgoingOffset_.push_back( currSize );
			outgoingSchedule_.push_back( tick );
			outgoingSize_.push_back( size );
			currSize += size;
			outgoingEntireSize_[tick] = currSize;
			if ( outgoing_.size() < currSize )
				outgoing_.resize( currSize );
			Field me( this, "destIn" );
			src.add( me );
}
void PostMasterWrapper::addIncomingFuncLocal( Field dest, int tick, int size )
{
//			cout << "Postmaster on node " << myNode_ << 
//					" got addIncomingFuncLocal( " << dest.path() <<
//					", " << tick << ", " << size << ")\n";

		countTicks();
			unsigned int currSize = incomingEntireSize_[tick];
			incomingSchedule_.push_back( tick );
			incomingSize_.push_back( size );
			currSize += size;
			incomingEntireSize_[tick] = currSize;
			if ( incoming_.size() < currSize )
				incoming_.resize( currSize );
			Field me( this, "srcOut" );
			me.add( dest );
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

Element* remoteCommandConnPostMasterLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( PostMasterWrapper, remoteCommandConn_ );
	return reinterpret_cast< PostMasterWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
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
		SolverConn* sc = dynamic_cast< SolverConn* >( c );
		if ( sc )
			return pm->getPostPtr( sc->index() );
	}
	return dummy;
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
void PostMasterWrapper::checkPendingRequests()
{
	if ( asyncRequest_.Test( status_ ) ) {
		int size = status_.Get_count( MPI_CHAR );
		if ( size < ASYNC_BUF_SIZE )
			asyncBuf_[size] = '\0';
		string temp( asyncBuf_ );

		// cout << "IRECV@" << myNode_ << " from " << remoteNode_ << 
		//	 ": " << temp << endl;

		remoteCommandSrc_.send( temp );

		// Post a recv for the next message.
		asyncRequest_ = comm_.Irecv( 
			asyncBuf_, ASYNC_BUF_SIZE, MPI_CHAR, remoteNode_, ASYNC_TAG );
	}
}
