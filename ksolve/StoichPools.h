/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _STOICH_POOLS_H
#define _STOICH_POOLS_H

class StoichPools
{
	public: 
		StoichPools();
		~StoichPools();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		/// Using the computed array sizes, now allocate space for them.
		void resizeArrays( unsigned int totNumPools );

		/**
		 * Assign spatial grid of pools
		 */
		void meshSplit( 
						vector< double > initConc,
						vector< double > vols,
						vector< unsigned int > localEntryList );


		/**
		 * Returns the vector of doubles of current mol #s at the specified
		 * mesh index
		 */
		const double* S( unsigned int meshIndex ) const;

		/**
		 * Returns the vector of doubles of current mol #s at the specified
		 * mesh index. Dangerous, allows one to modify the values.
		 */
		double* varS( unsigned int meshIndex );

		/**
		 * Returns the vector of doubles of initial mol #s at the specified
		 * mesh index
		 */
		const double* Sinit( unsigned int meshIndex ) const;

		//////////////////////////////////////////////////////////////////
		// Field assignment functions
		//////////////////////////////////////////////////////////////////

		void innerSetN( unsigned int meshIndex, 
						unsigned int poolIndex, double v );
		void innerSetNinit( unsigned int meshIndex, 
						unsigned int poolIndex, double v );
	private:
		/**
		 * 
		 * S_ is the array of molecules. Stored as n, number of molecules
		 * per mesh entry. 
		 * The array looks like n = S_[meshIndex][poolIndex]
		 * The meshIndex specifies which spatial mesh entry to use.
		 * The poolIndex specifies which molecular species pool to use.
		 * We choose the poolIndex as the right-hand index because we need
		 * to be able to pass the entire block of pools around for 
		 * integration.
		 *
		 * The entire S_ vector is allocated, but the pools are only 
		 * allocated for local meshEntries and for pools on
		 * diffusive boundaries with other nodes.
		 *
		 * The first numVarPools_ in the poolIndex are state variables and
		 * are integrated using the ODE solver. 
		 * The last numEfflux_ molecules within numVarPools are those that
		 * go out to another solver. They are also integrated by the ODE
		 * solver, so that at the end of dt each has exactly as many
		 * molecules as diffused away.
		 * The next numBufPools_ are fixed but can be changed by the script.
		 * The next numFuncPools_ are computed using arbitrary functions of
		 * any of the molecule levels, and the time.
		 * The functions evaluate _before_ the ODE. 
		 * The functions should not cascade as there is no guarantee of
		 * execution order.
		 */
		vector< vector< double > > S_;

		/**
		 * Sinit_ specifies initial conditions at t = 0. Whenever the reac
		 * system is rebuilt or reinited, all S_ values become set to Sinit.
		 * Also used for buffered molecules as the fixed values of these
		 * molecules.
		 * The array looks like Sinit_[meshIndex][poolIndex]
		 * The entire Sinit_ vector is allocated, but the pools are only 
		 * allocated for local meshEntries and for pools on
		 * diffusive boundaries with other nodes.
		 */
		vector< vector< double > > Sinit_;

		/**
		 * vector of indices of meshEntries to be computed locally.
		 * This may not necessarily be a contiguous set, depending on
		 * how boundaries and voxelization is done.
		 * globalMeshIndex = localMeshEntries_[localIndex]
		 */
		vector< unsigned int > localMeshEntries_;
};

#endif	// _STOICH_POOLS_H
