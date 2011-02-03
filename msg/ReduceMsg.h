/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _REDUCE_MSG_H
#define _REDUCE_MSG_H

/**
 * This is a specialized msg type used during field set and gets. It is
 * always anchored at e1 to the Shell, and at e2 to the actual target.
 */

class ReduceMsg: public Msg
{
	friend void initMsgManagers(); // for initializing Id.
	public:
		ReduceMsg( Eref e1, Element* e2, const ReduceFinfoBase* rfb );
		~ReduceMsg();

		void exec( const char* arg, const ProcInfo* p) const;

		Id id() const;

		FullId findOtherEnd( FullId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;

		void addToQ( const Element* src, Qinfo& q, const ProcInfo* p,
			MsgFuncBinding i, const char* arg ) const;
	private:
		DataId i1_;

		/**
		 * Pointer to source ReduceFinfoBase field on source Element.
		 * This keeps track of typing and creation of the correct
		 * Reduce subclass object.
		 */
		const ReduceFinfoBase* rfb_;

		static Id id_;
};

#endif // _REDUCE_MSG_H
