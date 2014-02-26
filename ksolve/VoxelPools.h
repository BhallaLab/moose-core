/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _VOXEL_POOLS_H
#define _VOXEL_POOLS_H

class Stoich;
class VoxelPools
{
	public: 
		VoxelPools();
		virtual ~VoxelPools();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		/// Allocate # of pools
		void resizeArrays( unsigned int totNumPools );

		/**
		 * Returns # of pools
		 */
		unsigned int size() const;

		/**
		 * Returns the array of doubles of current mol #s at the specified
		 * mesh index
		 */
		const double* S() const;

		/**
		 * Returns the array of doubles of current mol #s at the specified
		 * mesh index. Dangerous, allows one to modify the values.
		 */
		double* varS();

		/**
		 * Returns the array of doubles of initial mol #s at the specified
		 * mesh index
		 */
		const double* Sinit() const;

		/**
		 * Returns the array of doubles of initial mol #s at the specified
		 * mesh index, as a writable array.
		 */
		double* varSinit();

		/**
		 * Assigns S = Sinit.
		 */
		void reinit();

		//////////////////////////////////////////////////////////////////
		// Solver interface functions
		//////////////////////////////////////////////////////////////////
		void setStoich( const Stoich* stoich, const OdeSystem* ode );
		void advance( const ProcInfo* p );
		
		//////////////////////////////////////////////////////////////////
		// Field assignment functions
		//////////////////////////////////////////////////////////////////

		void setN( unsigned int i, double v );
		double getN( unsigned int ) const;
		void setNinit( unsigned int, double v );
		double getNinit( unsigned int ) const;
		void setDiffConst( unsigned int, double v );
		double getDiffConst( unsigned int ) const;

	private:
		/**
		 * 
		 * S_ is the array of molecules. Stored as n, number of molecules
		 * per mesh entry. 
		 * The poolIndex specifies which molecular species pool to use.
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
		vector< double > S_;

		/**
		 * Sinit_ specifies initial conditions at t = 0. Whenever the reac
		 * system is rebuilt or reinited, all S_ values become set to Sinit.
		 * Also used for buffered molecules as the fixed values of these
		 * molecules.
		 */
		vector< double > Sinit_;
#ifdef USE_GSL
		gsl_odeiv2_driver* driver_;
#endif
};

#endif	// _VOXEL_POOLS_H
