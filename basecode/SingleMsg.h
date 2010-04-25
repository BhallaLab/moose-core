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
	public:
		SingleMsg( Eref e1, Eref e2 );
		~SingleMsg() {;}

		void exec( const char* arg, const ProcInfo* p) const;

		bool isMsgHere( const Qinfo& q ) const;

		static bool add( Eref e1, const string& srcField, 
			Eref e2, const string& destField );

	private:
		DataId i1_;
		DataId i2_;
};

#endif // _SINGLE_MSG_H
