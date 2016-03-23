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

static size_t const     cacheline_size = 64;
typedef char            cacheline_pad_t [cacheline_size];

#define RKF45_ODEIV_FN_EVAL(S,t,y,f )   (*((S)->function))(t,y,f,(S)->params)

 typedef struct
{

	   double *k1;
	   double *k2;
	   double *k3;
	   double *k4;
	   double *k5;
	   double *k6;
	   double *y0;
	   double *ytmp;
}
rkf45_state_t;

/* Runge-Kutta-Fehlberg coefficients. Zero elements left out */

static const double ah[] = { 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0 };
static const double b3[] = { 3.0/32.0, 9.0/32.0 };
static const double b4[] = { 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0};
static const double b5[] = { 8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0};
static const double b6[] = { -6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0};

static const double c1 = 902880.0/7618050.0;
static const double c3 = 3953664.0/7618050.0;
static const double c4 = 3855735.0/7618050.0;
static const double c5 = -1371249.0/7618050.0;
static const double c6 = 277020.0/7618050.0;

static const double ec[] = { 0.0,
1.0 / 360.0,
0.0,
-128.0 / 4275.0,
-2197.0 / 75240.0,
1.0 / 50.0,
2.0 / 55.0
};


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
		void reinit( double dt );
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
		void advance( const ProcInfo* p);

		//RungeKutta functions by Rahul
		int rungeKutta45(void *vstate, size_t dim, double t, double h, double* y, double* yerr, const double* dydt_in, double* dydt_out, const gsl_odeiv2_system * sys);

		/// Set initial timestep to use by the solver.
		void setInitDt( double dt );

		/// This is the function which evaluates the rates.
		static int gslFunc( double t, const double* y, double *dydt, 
						void* params );

		//////////////////////////////////////////////////////////////////
		// Rate manipulation and calculation functions
		//////////////////////////////////////////////////////////////////
		/// Handles volume change and subsequent cascading updates.
		void setVolumeAndDependencies( double vol );

		/// Updates all the rate constants from the reference rates vector.
		void updateAllRateTerms( const vector< RateTerm* >& rates,
					   unsigned int numCoreRates	);
		/**
		 * updateRateTerms updates the rate consts of a belonging to 
		 * the specified index on the rates vector.
		 * It does recaling and assigning using values from the 
		 * internal rates vector.
		 */
		void updateRateTerms( const vector< RateTerm* >& rates,
			unsigned int numCoreRates, unsigned int index );

		/**
		 * Core computation function. Updates the reaction velocities
		 * vector yprime given the current mol 'n' vector s.
		 */
		void updateRates( const double* s, double* yprime ) const;

		/**
		 * updateReacVelocities computes the velocity *v* of each reaction
		 * from the vector *s* of pool #s.
		 * This is a utility function for programs like SteadyState that 
		 * need to analyze velocity.
		 */
		void updateReacVelocities( 
						const double* s, vector< double >& v ) const;

		gsl_odeiv2_driver* getVoxeldriver(){return driver_;}
		gsl_odeiv2_system getVoxelsys(){return sys_;}

		/**
		 * Changes cross rate terms to zero if there is no junction
		void filterCrossRateTerms( const vector< pair< Id, Id > >& vec );
		 */

		/// Used for debugging.
		void print() const;
	private:
#ifdef USE_GSL
		gsl_odeiv2_system sys_;
		gsl_odeiv2_driver* driver_;

//		int GSLSUCCESS;
//		cacheline_pad_t pad;
#endif
};

#endif	// _VOXEL_POOLS_H
