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


// Dummy functions
void isend( char* buf, int size, char* name, int dest );
void irecv( char* buf, int size, char* name, int src );

#include "header.h"
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
		"remoteNode", &PostMasterWrapper::getRemoteNode, "int" ),
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

void PostMasterWrapper::processFuncLocal( ProcInfo info )
{
	long index = 0;
	isend( &(outgoing_.front()) + outgoingBufferOffset_[ index ],
		outgoingBufferSize_[ index ], "char", remoteNode_);
	for (size_t i = 0; i < outgoing_.size(); i++ )
		incoming_[i] = outgoing_[i];
	irecv( &(incoming_.front()) + incomingBufferOffset_[ index ],
		incomingBufferSize_[ index ], "char", remoteNode_);
	srcSrc_.send( &( incoming_.front() ) );
}

void PostMasterWrapper::reinitFuncLocal(  )
{
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
			destInConn_.resize( segments );
			// processInConn_.resize( segments );
			if ( outgoingSchedule_.size() > 0 ) {
				outgoingBufferOffset_.resize( outgoingSchedule_.size());
				outgoingBufferSize_.resize( outgoingSchedule_.size() );
				outgoingBufferOffset_[0] = 0;
				outgoingBufferSize_[0] = offset;
			}
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
			srcOutConn_( this ),
			destInConn_( this )
			// processInConn_( this )
{
	vector< unsigned long > segments( 1, 0 );
	segments[0] = 10;
	destInConn_.resize( segments );
	// processInConn_.resize( segments );
	outgoingOffset_.resize( 1, 0 );
	outgoing_.resize( 40 );
	outgoingBufferOffset_.resize( 1, 0 );
	outgoingBufferSize_.resize( 1, 0 );
	incomingOffset_.resize( 1, 0 );
	incoming_.resize( 40 );
	incomingBufferOffset_.resize( 1, 0 );
	incomingBufferSize_.resize( 1, 0 );
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
		SolveMultiConn* mc = dynamic_cast< SolveMultiConn* >( c );
		SolverConn* sc = dynamic_cast< SolverConn* >( c );
		if ( sc )
			return pm->getPostPtr( sc->index() );
	}
	return dummy;
}
void isend( char* buf, int size, char* name, int dest )
{
	cout << "in isend, with " << size << " bytes\n";

}
void irecv( char* buf, int size, char* name, int src )
{
	cout << "in irecv, expecting " << size << " bytes\n";
}
void PostMaster::addSender( const Finfo* sender )
{
	outgoingSize_.push_back( sender->ftype()->size() );
	outgoingSchedule_.push_back( 0 );
	outgoingOffset_.push_back( 
		outgoingOffset_.back() + outgoingSize_.back() );
}
