/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FUNC_BASE_H
#define _FUNC_BASE_H

/**
 * FuncBase.
 * This is the MOOSE base class for doing arbitrary functions.
 * Just holds and operates a FuncTerm class, which is a generic 
 * class for doing functions. Designed so it doesn't need zombification.
 * Instead the Stoich just uses the FuncTerm defined and allocated
 * in the derived class of a FuncBase.
 */
class FuncBase {
	public:
		FuncBase();
		virtual ~FuncBase();
		void process( const Eref& e, ProcPtr info);
		void reinit( const Eref& e, ProcPtr info );
		void input( double d );
		double getResult() const;

		virtual void vProcess( const Eref& e, ProcPtr info ) = 0;
		virtual void vReinit( const Eref& e, ProcPtr info ) = 0;
		virtual void vInput( double d ) = 0;
		virtual FuncTerm* func() = 0;
		
		static const Cinfo* initCinfo();
	protected:
		double result_;
};

#endif // _FUNC_BASE_H
