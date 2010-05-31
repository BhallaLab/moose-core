/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ONE_TO_ONE_MSG_H
#define _ONE_TO_ONE_MSG_H

/**
 * Manages a projection where each entry in source array
 * connects to the corresponding entry (with same index)
 * in dest array.
 */
class OneToOneMsg: public Msg
{
	friend void initMsgManagers();
	public:
		OneToOneMsg( Element* e1, Element* e2 );
		~OneToOneMsg();

		void exec( const char* arg, const ProcInfo* p) const;

		Id id() const;

		FullId findOtherEnd( FullId end ) const;

		Msg* copy( Id origSrc, Id newSrc, Id newTgt,
			FuncId fid, unsigned int b, unsigned int n ) const;
	private:
		static Id id_;
};

// No extra fields compared to the MsgManager.
class OneToOneMsgWrapper: public MsgManager
{
	public:
		static const Cinfo* initCinfo();
	private:
		// MsgId mid_;
};

#endif // _ONE_TO_ONE_MSG_H
