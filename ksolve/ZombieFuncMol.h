/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_FUNC_MOL_H
#define _ZOMBIE_FUNC_MOL_H

class ZombieFuncMol: public ZombieMol
{
	public: 
		ZombieFuncMol();

		void input( double v );

		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _ZOMBIE_FUNC_MOL_H
