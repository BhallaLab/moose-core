#ifndef _ParTable_h
#define _ParTable_h

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


class ParTable: public Table
{
	public:
		ParTable();

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setIndex( const Conn& c, int value );
		static int getIndex( const Element* e );

		////////////////////////////////////////////////////////////
		// Here are the Table Destination functions
		////////////////////////////////////////////////////////////
		static void process( const Conn& c, ProcInfo p );
		static void reinit( const Conn& c, ProcInfo p );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		void innerProcess( Element* e, ProcInfo p );
		void innerReinit( const Conn& c, ProcInfo p );

	private:
		int index_;
		unsigned long ulTableIndex_;
		unsigned long ulCurrentIndex_;
		bool bSelected_;
		bool bRecvCalled_;
		bool bSendCalled_;
		double lRequest_;
		MPI_Request recv_request_;
		MPI_Request send_request_[MAX_MPI_RECV_RECORD_SIZE/VISLN_CHUNK_SIZE];
		MPI_Status status_;
		unsigned long ulLastRecordSent_;

};

extern const Cinfo* initParTableCinfo();

#endif // _ParTable_h
