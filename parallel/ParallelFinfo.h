/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _PARALLEL_FINFO_H
#define _PARALLEL_FINFO_H

class ParallelDummyFinfo: public Finfo {
	public:
		ParallelDummyFinfo(const string& name)
		: Finfo(name)
		{ ; }

		~ParallelDummyFinfo()
		{ ; }

		RecvFunc recvFunc( ) const {
			return rf_;
		}

		unsigned long matchRemoteFunc(
			Element* e, RecvFunc rf ) const {
			return 0;
		}

		void addRecvFunc( Element* e, RecvFunc rf,
			unsigned long position )
		{
			rf_ = rf;
		}

		void setInConn( Conn* inConn ) {
			inConn_ = inConn;
		}

		Conn* inConn( Element* ) const {
			return inConn_;
		}

		Conn* outConn( Element* ) const {
			return dummyConn();
		}


		// Here are the message creation and removal operations.
		bool add( Element* e, Field& destfield, bool useSharedConn ) {
			return 0;
		}
		
		// The destination of a message checks type of sender, and if
		// all is well returns a Finfo to use for the message.
		Finfo* respondToAdd( Element* e, const Finfo* sender ) { 
			return 0;
		}

		void initialize( const Cinfo* c ) {
			;
		}

		const Ftype* ftype() const;

		Finfo* makeRelayFinfo( Element* e ) {
			return this;
		}

	private:
		RecvFunc rf_;
		Conn* inConn_;
};

class ParallelDestFinfo: public Dest0Finfo
{
	public:
		ParallelDestFinfo( const string& name, 
			void ( *rFunc )( Conn* ),
			Conn* ( *getConn )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	Dest0Finfo( name, rFunc, getConn, triggers, sharesConn )
		{
			;
		}

		// This is the critical operation. The target PostMaster
		// will figure out where the message is actually going,
		// and accordingly position this message to dump data into
		// the correct buffer location. It will also scan back to
		// the sender and find out which clock tick is needed to
		// schedule the transmission.
		// At this level, all we do is return a dummy finfo with 
		// the sender's PostRecvFunc.
		Finfo* respondToAdd( Element* e, const Finfo* sender );
};

class ParallelSrcFinfo: public NSrc0Finfo
{
	public:
		ParallelSrcFinfo( const string& name, 
			NMsgSrc* ( *getSrc )( Element * ),
			const string& triggers,
			bool sharesConn = 0
		)
		:	NSrc0Finfo( name, getSrc, triggers, sharesConn ) 
		{
			;
		}

		bool add( Element* e, Field& destfield, bool useSharedConn = 0);
};

#endif // _PARALLEL_FINFO_H
