/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#define DATA_TAG 0
#define ASYNC_TAG 1

#define ASYNC_BUF_SIZE 1000


#ifndef _PostMaster_h
#define _PostMaster_h
class PostMaster
{
	friend class PostMasterWrapper;
	public:
		PostMaster()
			: needsReinit_( 0 ), 
			numTicks_(0),
			comm_( MPI::COMM_WORLD )
		{
			remoteNode_ = 0;
			myNode_ = MPI::COMM_WORLD.Get_rank();
			pollFlag_ = 0;
			asyncBuf_ = new char[ASYNC_BUF_SIZE];
		}
		void addSender( const Finfo* sender );

	private:
		int myNode_;
		int remoteNode_;
		bool needsReinit_;
		unsigned long numTicks_;
		int pollFlag_;
		MPI::Comm& comm_;
		MPI::Request request_;
		MPI::Request asyncRequest_;
		MPI::Status status_;
		vector< unsigned int > outgoingOffset_;
		vector< unsigned int > outgoingSchedule_;
		vector< unsigned int > outgoingSize_;
		vector< char > outgoing_;
		vector< unsigned int > outgoingEntireSize_;
		vector< unsigned int > incomingSchedule_;
		vector< unsigned int > incomingSize_;
		vector< unsigned int > incomingEntireSize_;
		vector< char > incoming_;
		char* asyncBuf_;
};
#endif // _PostMaster_h
