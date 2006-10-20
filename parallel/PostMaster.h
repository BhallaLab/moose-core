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

#ifndef _PostMaster_h
#define _PostMaster_h
class PostMaster
{
	friend class PostMasterWrapper;
	public:
		PostMaster()
		{
			remoteNode_ = 0;
			localNode_ = 0;
		}

		void addSender( const Finfo* f );

	private:
		int remoteNode_;
		int localNode_;
		vector< unsigned int > outgoingOffset_;
		vector< unsigned int > outgoingSchedule_;
		vector< unsigned int > outgoingSize_;
		vector< char > outgoing_;
		vector< unsigned int > outgoingBufferTime_;
		vector< unsigned int > outgoingBufferSize_;
		vector< unsigned int > outgoingBufferOffset_;
		vector< unsigned int > incomingOffset_;
		vector< unsigned int > incomingSchedule_;
		vector< char > incoming_;
		vector< unsigned int > incomingBufferTime_;
		vector< unsigned int > incomingBufferSize_;
		vector< unsigned int > incomingBufferOffset_;
};
#endif // _PostMaster_h
