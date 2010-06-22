/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_REAC_H
#define _ZOMBIE_REAC_H

class ZombieReac: public Stoich
{
	public: 
		ZombieReac();
		~ZombieReac();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setKf( Eref e, const Qinfo* q, double v );
		double getKf( Eref e, const Qinfo* q ) const;
		void setKb( Eref e, const Qinfo* q, double v );
		double getKb( Eref e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void eprocess( Eref e, const Qinfo* q, ProcInfo* p );
		void reinit( const Eref& e, const Qinfo*q, ProcInfo* p );
		void sub( double v );
		void prd( double v );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		unsigned int convertId ( Id id ) const;
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _ZOMBIE_REAC_H
