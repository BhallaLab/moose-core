/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_SUMFUNC_H
#define _ZOMBIE_SUMFUNC_H

/**
 * ZombieSumFunc.
 * This is the MOOSE class to sum together assorted inputs.
 */
class ZombieSumFunc: public Stoich
{
	public:
		ZombieSumFunc();
		void process( const Eref& e, ProcPtr info);
		void reinit( const Eref& e, ProcPtr info );
		void input( double d );
		double getResult( const Eref& e, const Qinfo* q ) const;

		static void zombify( Element* solver, Element* orig, Id molId );
		static void unzombify( Element* zombie );
		static const Cinfo* initCinfo();
	private:
		double result_;
};

#endif // _ZOMBIE_SUMFUNC_H
