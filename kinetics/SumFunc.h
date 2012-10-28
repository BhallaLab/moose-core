/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SUMFUNC_H
#define _SUMFUNC_H

/**
 * SumFunc.
 * This is the MOOSE class to sum together assorted inputs.
 */
class SumFunc: public FuncBase
{
	public:
		SumFunc();
		~SumFunc();
		void vProcess( const Eref& e, ProcPtr info);
		void vReinit( const Eref& e, ProcPtr info );
		void vInput( double d );
		FuncTerm* func();

		static const Cinfo* initCinfo();
	private:
		SumTotalTerm st_;
};

#endif // _SUMFUNC_H
