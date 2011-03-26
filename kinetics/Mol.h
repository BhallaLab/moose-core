/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MOL_H
#define _MOL_H

class Mol
{
	friend void testSyncArray( unsigned int size, unsigned int numThreads,
		unsigned int method );
	friend void checkVal( double time, const Mol* m, unsigned int size );
	friend void forceCheckVal( double time, Element* e, unsigned int size );

	public: 
		Mol();
		Mol( double nInit );

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

		void setSpecies( unsigned int v );
		unsigned int getSpecies() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void reac( double A, double B );

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

#endif	// _MOL_H
