/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GSSA_VOXEL_POOLS_BASE_H
#define _GSSA_VOXEL_POOLS_BASE_H

class Stoich;
class GssaVoxelPools: public VoxelPoolsBase
{
	public: 
		GssaVoxelPools();
		virtual ~GssaVoxelPools();


		//////////////////////////////////////////////////////////////////
		// Solver interface functions
		//////////////////////////////////////////////////////////////////
		void advance( const ProcInfo* p );
		void updateDependentMathExpn( 
				const GssaSystem* g, unsigned int rindex );
		void updateDependentRates( 
			const vector< unsigned int >& deps, const Stoich* stoich );
		unsigned int pickReac() const;
		void setNumReac( unsigned int n );

		void advance( const ProcInfo* p, const GssaSystem* g );

		/**
 		* Cleans out all reac rates and recalculates atot. Needed whenever a
 		* mol conc changes, or if there is a roundoff error. Returns true
 		* if OK, returns false if it is in a stuck state and atot<=0
 		*/
		bool refreshAtot( const GssaSystem* g );

		/**
		 * Builds the gssa system as needed.
		 */
		void reinit( const GssaSystem* g );

		void updateAllRateTerms( const vector< RateTerm* >& rates,
					   unsigned int numCoreRates	);
		void updateRateTerms( const vector< RateTerm* >& rates,
			unsigned int numCoreRates, unsigned int index );

		double getReacVelocity( unsigned int r, const double* s ) const;
		void updateReacVelocities( const GssaSystem* g,
			const double* s, vector< double >& v ) const;

		/**
		 * Assign the volume, and handle the cascading effects by scaling
		 * all the dependent values of nInit and rates if applicable.
		 */
		void setVolumeAndDependencies( double vol );

		void setStoich( const Stoich* stoichPtr );

	private:
		const Stoich* stoichPtr_;

		/// Time at which next event will occur.
		double t_; 

		/**
		 * Total propensity of all the reactions in the system
		 */
		double atot_;

		/** 
		 * State vector of reaction velocities. Only a subset are
		 * recalculated on each step.
		 */
		vector< double > v_; 

		// Possibly we should put independent RNGS, so save one here.
		
		/// Looks up vol-matched Rates from the stoich
		//unsigned int volIndex_;

		vector< RateTerm* > rates_;
};

#endif	// _GSSA_VOXEL_POOLS_H
