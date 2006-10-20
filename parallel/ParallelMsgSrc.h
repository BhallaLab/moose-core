/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _PARALLEL_MSGSRC_H
#define _PARALLEL_MSGSRC_H


// We need one of these for each scheduled tick.
class ParallelMsgSrc: public NMsgSrc
{
	public:
		ParallelMsgSrc( BaseMultiConn* c )
			: NMsgSrc( c )
		{
			;
		}

		bool add( RecvFunc rf, const Ftype* ft, Conn* target );
		void send( char* dataPtr );

	private:
		vector< const Ftype* >targetType_;
};
#endif	// _PARALLEL_MSGSRC_H
