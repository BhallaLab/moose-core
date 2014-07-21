/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_POOL_INTERFACE_H
#define _ZOMBIE_POOL_INTERFACE_H

/**
 * This pure virtual base class is for solvers that want to talk to
 * the zombie pool. 
 * The Eref specifies both the pool identity and the voxel number within
 * the pool.
 */
class ZombiePoolInterface
{
	public:
		/// Set initial # of molecules in given pool and voxel. Bdry cond.
		virtual void setNinit( const Eref& e, double val ) = 0;
		/// get initial # of molecules in given pool and voxel. Bdry cond.
		virtual double getNinit( const Eref& e ) const = 0;

		/// Set # of molecules in given pool and voxel. Varies with time.
		virtual void setN( const Eref& e, double val ) = 0;
		/// Get # of molecules in given pool and voxel. Varies with time.
		virtual double getN( const Eref& e ) const = 0;

		/// Diffusion constant: Only one per pool, voxel number is ignored.
		virtual void setDiffConst( const Eref& e, double val ) = 0;
		/// Diffusion constant: Only one per pool, voxel number is ignored.
		virtual double getDiffConst( const Eref& e ) const = 0;

		/// Motor constant: Only one per pool, voxel number is ignored.
		/// Used only in Dsolves, so here I put in a dummy.
		virtual void setMotorConst( const Eref& e, double val )
		{;}

		/// Specifies number of pools (species) handled by system.
		virtual void setNumPools( unsigned int num ) = 0;
		/// gets number of pools (species) handled by system.
		virtual unsigned int getNumPools() const = 0;

		/**
		 * Gets block of data. The first 4 entries are passed in 
		 * on the 'values' vector: the start voxel, numVoxels, 
		 * start pool#, numPools.
		 * These are followed by numVoxels * numPools of data values
		 * which are filled in by the function.
		 * We assert that the entire requested block is present in 
		 * this ZombiePoolInterface.
		 * The block is organized as an array of arrays of voxels;
		 * values[pool#][voxel#]
		 *
		 * Note that numVoxels and numPools are the number in the current
		 * block, not the upper limit of the block. So 
		 * values.size() == 4 + numPools * numVoxels.
		 */
		virtual void getBlock( vector< double >& values ) const = 0;

		/**
		 * Sets block of data. The first 4 entries 
		 * on the 'values' vector are the start voxel, numVoxels, 
		 * start pool#, numPools. These are 
		 * followed by numVoxels * numPools of data values.
		 */
		virtual void setBlock( const vector< double >& values ) = 0;

		/**
		 * Informs the ZPI about the stoich, used during subsequent
		 * computations.
		 * Called to wrap up the model building. The Stoich
		 * does this call after it has set up its own path.
		 */
		virtual void setStoich( Id stoich ) = 0;

		/// Assignes the diffusion solver. Used by the reac solvers
		virtual void setDsolve( Id dsolve ) = 0;

		/// Assigns compartment.
		virtual void setCompartment( Id compartment ) = 0;
		virtual Id getCompartment() const = 0;

		/// Sets up cross-solver reactions.
		virtual void setupCrossSolverReacs( 
			const map< Id, vector< Id > >& xr, 
			Id otherStoich ) = 0;
};

#endif	// _ZOMBIE_POOL_INTERFACE_H
