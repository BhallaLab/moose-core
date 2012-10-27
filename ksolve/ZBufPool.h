/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Z_BUF_POOL_H
#define _Z_BUF_POOL_H

class ZBufPool: public ZPool
{
	public: 
		ZBufPool();
		~ZBufPool();

		/// The 'get' functions are simply inherited from ZPool
		void vSetN( const Eref& e, const Qinfo* q, double v );
		void vSetNinit( const Eref& e, const Qinfo* q, double v );
		void vSetConc( const Eref& e, const Qinfo* q, double v );
		void vSetConcInit( const Eref& e, const Qinfo* q, double v );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _Z_BUF_POOL_H
