/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _GROUP_H
#define _GROUP_H

class Group: public Data
{
	public: 
		Group();
		void process( const ProcInfo* p, const Eref& e );

		static const Cinfo* initCinfo();
	private:
};

#endif // _GROUP_H
