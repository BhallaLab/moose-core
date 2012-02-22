/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_BUF_POOL_H
#define _ZOMBIE_BUF_POOL_H

class ZombieBufPool: public ZombiePool
{
	public: 
		ZombieBufPool();

		void setN( const Eref& e, const Qinfo* q, double v );
		double getN( const Eref& e, const Qinfo* q ) const;
		void setNinit( const Eref& e, const Qinfo* q, double v );
		double getNinit( const Eref& e, const Qinfo* q ) const;

		void setConc( const Eref& e, const Qinfo* q, double v );
		double getConc( const Eref& e, const Qinfo* q ) const;
		void setConcInit( const Eref& e, const Qinfo* q, double v );
		double getConcInit( const Eref& e, const Qinfo* q ) const;

		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _ZOMBIE_BUF_POOL_H
