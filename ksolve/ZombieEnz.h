/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_ENZ_H
#define _ZOMBIE_ENZ_H

class ZombieEnz: public Stoich
{
	public: 
		ZombieEnz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setK1( const Eref& e, const Qinfo* q, double v );
		double getK1( const Eref& e, const Qinfo* q ) const;
		void setK2( const Eref& e, const Qinfo* q, double v );
		double getK2( const Eref& e, const Qinfo* q ) const;
		void setK3( const Eref& e, const Qinfo* q, double v );
		double getK3( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void dummy( double n );

		//////////////////////////////////////////////////////////////////
		// Utility  funcs
		//////////////////////////////////////////////////////////////////
		ZeroOrder* makeHalfReaction( 
			Element* orig, double rate, const SrcFinfo* finfo, Id enz )
			const;
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif // _ZOMBIE_ENZ_H
