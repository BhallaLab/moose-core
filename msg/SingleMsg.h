/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SINGLE_MSG_H
#define _SINGLE_MSG_H


class SingleMsg: public Msg
{
	friend void initMsgManagers(); // for initializing Id.
	public:
		SingleMsg( Eref e1, Eref e2 );
		~SingleMsg();

		void exec( const char* arg, const ProcInfo* p) const;

		bool isMsgHere( const Qinfo& q ) const;

		DataId i1() const;
		DataId i2() const;

		void setI1( DataId di );
		void setI2( DataId di );

		// returns the id of the managing Element.
		Id id() const;

		FullId findOtherEnd( FullId end ) const;

	private:
		static void setId( Id id );
		DataId i1_;
		DataId i2_;
		static Id id_;
};

class SingleMsgWrapper: public MsgManager
{
	public:
		/*
		Id getE1() const;
		Id getE2() const;
		*/

		void setI1( DataId di );
		DataId getI1() const;

		void setI2( DataId di );
		DataId getI2() const;

		static const Cinfo* initCinfo();
	private:
		// MsgId mid_;
};

#endif // _SINGLE_MSG_H
