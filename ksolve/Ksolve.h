/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _KSOLVE_H
#define _KSOLVE_H

class Stoich;
class Ksolve: public ZombiePoolInterface
{
	public: 
		Ksolve();
		~Ksolve();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		Id getStoich() const;
		void setStoich( Id stoich );

		unsigned int getNumLocalVoxels() const;
		unsigned int getNumAllVoxels() const;
		//////////////////////////////////////////////////////////////////
		// Dest Finfos
		//////////////////////////////////////////////////////////////////
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		//////////////////////////////////////////////////////////////////
		// Solver interface functions
		//////////////////////////////////////////////////////////////////
		unsigned int getPoolIndex( const Eref& e ) const;
		unsigned int getVoxelIndex( const Eref& e ) const;
		
		//////////////////////////////////////////////////////////////////
		// ZombiePoolInterface inherited functions
		//////////////////////////////////////////////////////////////////

		void setN( const Eref& e, double v );
		double getN( const Eref& e ) const;
		void setNinit( const Eref& e, double v );
		double getNinit( const Eref& e ) const;
		void setDiffConst( double v );
		double getDiffConst() const;

		static const Cinfo* initCinfo();
	private:
		/**
		 * Each VoxelPools entry handles all the pools in a single voxel.
		 * Each entry knows how to update itself in order to complete 
		 * the kinetic calculations for that voxel. The ksolver does
		 * multinode management by indexing only the subset of entries
		 * present on this node.
		 */
		vector< VoxelPools > pools_;

		/// First voxel indexed on the current node.
		unsigned int startVoxel_;

		/**
		 * Stoich is the class that sets up the reaction system and
		 * manages the stoichiometry matrix
		 */
		Id stoich_;

		/// Utility ptr used to help Pool Id lookups by the Ksolve.
		const Stoich* stoichPtr_;
};

#endif	// _KSOLVE_H
