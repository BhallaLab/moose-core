/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_TO_ALL_MSG_H
#define _ONE_TO_ALL_MSG_H

/**
 * Manages projection from a single entry in e1 to all 
 * array entries in e2.
 */

class OneToAllMsg: public Msg
{
	friend void initMsgManagers();
	public:
		OneToAllMsg( MsgId mid, Eref e1, Element* e2 );
		~OneToAllMsg();

		void exec( const char* arg, const ProcInfo* p ) const;


		bool isMsgHere( const Qinfo& q ) const;

		Id id() const;

		FullId findOtherEnd( FullId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;

		void addToQ( const Element* src, Qinfo& q, const ProcInfo* p,
			MsgFuncBinding i, const char* arg ) const;

		/// Return the first DataId
		DataId getI1() const;

		/// Setup function for Element-style access to Msg fields.
		static const Cinfo* initCinfo();
	private:
		DataId i1_;
		static Id id_;
};


#endif // _ONE_TO_ALL_MSG_H
