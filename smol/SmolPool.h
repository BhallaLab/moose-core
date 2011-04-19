/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_POOL_H
#define _SMOL_POOL_H

class SmolPool
{
	public: 
		SmolPool();
		~SmolPool();

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

		/**
		 * Initializes molecular species into Smoldyn
		 */
		static void smolSpeciesInit( Element* solver, Element* orig );

		/**
		 * Adds up nInit for each species so we can set the max # of 
		 * molecules in the simulation.
		 */
		static void smolMaxNumMolecules( simptr sim, const vector< Id >& pools );

		/**
		 * Initializes initial # of molecules of specified species
		 */
		static void smolNinit( Element* solver, Element* orig );

		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
		struct simstruct* sim_; 
		// The sim_ struct is always on the local node, so a ptr is OK.
		double nInit_;
		double diffConst_;
};

#endif	// _SMOL_POOL_H
