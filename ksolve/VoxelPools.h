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
/**
 * This is the class for handling reac-diff voxels used for deterministic
 * computations.
 */
class VoxelPools: public VoxelPoolsBase
{
	public: 
		VoxelPools();
		virtual ~VoxelPools();

		//////////////////////////////////////////////////////////////////
		// Solver interface functions
		//////////////////////////////////////////////////////////////////
		/**
		 * setStoich: Assigns the ODE system and Stoich. 
		 * Stoich may be modified to create a new RateTerm vector
		 * in case the volume of this VoxelPools is new.
		 */
		void setStoich( Stoich* stoich, const OdeSystem* ode );

		/// Do the numerical integration. Advance the simulation.
		void advance( const ProcInfo* p );

		/// Set initial timestep to use by the solver.
		void setInitDt( double dt );

		/// This is the function which evaluates the rates.
		static int gslFunc( double t, const double* y, double *dydt, 
						void* params );

	private:
		/// Used to identify which Rates_ vector to use for this volume.
		unsigned int volIndex_;
		const Stoich* stoichPtr_;
#ifdef USE_GSL
		gsl_odeiv2_driver* driver_;
		gsl_odeiv2_system sys_;
#endif
};

#endif	// _VOXEL_POOLS_H
