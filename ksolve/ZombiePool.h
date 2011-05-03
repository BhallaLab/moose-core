/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_POOL_H
#define _ZOMBIE_POOL_H

class ZombiePool: public Stoich
{
	public: 
		ZombiePool();
		~ZombiePool();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setN( const Eref& e, const Qinfo* q, double v );
		double getN( const Eref& e, const Qinfo* q ) const;
		void setNinit( const Eref& e, const Qinfo* q, double v );
		double getNinit( const Eref& e, const Qinfo* q ) const;
		void setDiffConst( const Eref& e, const Qinfo* q, double v );
		double getDiffConst( const Eref& e, const Qinfo* q ) const;

		void setConc( const Eref& e, const Qinfo* q, double v );
		double getConc( const Eref& e, const Qinfo* q ) const;
		void setConcInit( const Eref& e, const Qinfo* q, double v );
		double getConcInit( const Eref& e, const Qinfo* q ) const;

		void setSize( const Eref& e, const Qinfo* q, double v );
		double getSize( const Eref& e, const Qinfo* q ) const;

		void setSpecies( const Eref& e, const Qinfo* q, unsigned int v );
		unsigned int getSpecies( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void reac( double A, double B );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _ZOMBIE_POOL_H
