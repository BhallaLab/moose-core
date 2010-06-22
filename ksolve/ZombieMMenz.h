/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_MM_ENZ_H
#define _ZOMBIE_MM_ENZ_H

/**
 * This class represents the Michaelis-Menten type enzyme, obeying the
 * equation
 * V = kcat.[Etot].[S]/( Km + [S] )
 */
class ZombieMMenz: public Stoich
{
	public: 
		ZombieMMenz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setKm( Eref e, const Qinfo* q, double v );
		double getKm( Eref e, const Qinfo* q ) const;
		void setKcat( Eref e, const Qinfo* q, double v );
		double getKcat( Eref e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void eprocess( Eref e, const Qinfo* q, ProcInfo* p );
		void reinit( const Eref& e, const Qinfo*q, ProcInfo* p );
		void dummy( double n );

		//////////////////////////////////////////////////////////////////
		// Utility  funcs
		//////////////////////////////////////////////////////////////////
		unsigned int convertId ( Id id ) const;
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif // _ZOMBIE_MM_ENZ_H
