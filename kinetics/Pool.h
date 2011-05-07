/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _POOL_H
#define _POOL_H

/**
 * SpeciesId identifies molecular species. This is a unique identifier for
 * any given molecular species, regardless of which compartment or solver
 * is handling it. 
 */
typedef unsigned int SpeciesId;
extern const SpeciesId DefaultSpeciesId;

/**
 * The Pool class is a molecular pool. This is a set of molecules of a 
 * given species, in a uniform chemical context. Note that the same
 * species might be present in other compartments, or be handled by
 * other solvers.
 */
class Pool
{
	friend void testSyncArray( unsigned int size, unsigned int numThreads,
		unsigned int method );
	friend void checkVal( double time, const Pool* m, unsigned int size );
	friend void forceCheckVal( double time, Element* e, unsigned int size );

	public: 
		Pool();
		Pool( double nInit );

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setN( double v );
		double getN() const;
		void setNinit( double v );
		double getNinit() const;
		void setDiffConst( double v );
		double getDiffConst() const;

		void setConc( double v );
		double getConc() const;
		void setConcInit( double v );
		double getConcInit() const;

		/**
		 * Size is usually volume, but we also permit areal density
		 * This should be slaved to the parent compartment and is not
		 * meant to be set through field operations.
		 */
		void setSize( double v );
		double getSize() const;

		void setSpecies( SpeciesId v );
		SpeciesId getSpecies() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void handleMolWt( double v );
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void reac( double A, double B );
		void increment( double val );
		void decrement( double val );

		static const Cinfo* initCinfo();
	private:
		double n_; /// Number of molecules in pool
		double nInit_; /// initial condition 
		double size_;	/// volume/areal density of molecule.
		double diffConst_;	/// Diffusion constant
		double A_; /// Internal state variables, used only in explict mode
		double B_;

		/**
		 * System wide identifier for all mol pools that are chemically
		 * the same species.
		 */
		unsigned int species_; 
		
};

#endif	// _POOL_H
