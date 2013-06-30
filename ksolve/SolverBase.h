/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SOLVER_BASE_H
#define _SOLVER_BASE_H

#ifndef _CHEM_COMPT_H
class ChemCompt;
#endif
class SolverBase
{
	public: 
		SolverBase();
		virtual ~SolverBase();

		///////////////////////////////////////////////////
		// Zombie setup functions.
		///////////////////////////////////////////////////
		virtual void installReaction( Id reacId, 
				const vector< Id >& subs, const vector< Id >& prds ) = 0;
		virtual void installMMenz( Id enzId, Id enzMolId, 
				const vector< Id >& subs, const vector< Id >& prds ) = 0;
		virtual void installEnzyme( Id enzId, Id enzMolId, Id cplxId,
				const vector< Id >& subs, const vector< Id >& prds ) = 0;

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////

		/**
		 * Looks for the pools participating in cross-solver reactions,
		 * where the reactions are on the current Solver, and the pool is
		 * on the other solver. As a side-effect, sets up the local
		 * PoolIndices of this set of pools as the remoteReacPools on the
		 * current junction.
		 */
		virtual void findPoolsOnOther( Id other, vector< Id >& pools ) = 0;

		/**
		 * Takes the provided list of pools which go to a reaction on
		 * the other solver participating in the latest junction. Figures
		 * out poolIndices and assigns to junction.
		 */
		virtual void setLocalCrossReactingPools( 
						const vector< Id >& pools ) = 0;

		/**
		 * Converts pool Id to an internal index used in the solver.
		 * Exposes internal indexing for pool data, but useful for the 
		 * reactions in Gsl and GSSA modes.
		 */
		virtual unsigned int convertIdToPoolIndex( Id id ) const = 0;

		/**
		 * Callback function which tells the solver how many pools to build
		 */
		virtual void allocatePools( unsigned int numPools ) = 0;
		//////////////////////////////////////////////////////////////////
		// Field assignment functions for pools
		//////////////////////////////////////////////////////////////////

		virtual void setN( const Eref& e, double v ) = 0;
		virtual double getN( const Eref& e ) const = 0;
		virtual void setNinit( const Eref& e, double v ) = 0;
		virtual double getNinit( const Eref& e ) const = 0;
		virtual void setSpecies( const Eref& e, unsigned int v ) = 0;
		virtual unsigned int getSpecies( const Eref& e ) = 0;
		virtual void setDiffConst( const Eref& e, double v ) = 0;
		virtual double getDiffConst( const Eref& e ) const = 0;

		//////////////////////////////////////////////////////////////////
		// Pure virtual Field access functions for reacs and enz.
		//////////////////////////////////////////////////////////////////

		// Assignment of Kf and Kb are in conc units.
		virtual void setReacKf( const Eref& e, double v ) const = 0;
		virtual void setReacKb( const Eref& e, double v ) const = 0;
		virtual double getReacNumKf( const Eref& e ) const = 0;
		virtual double getReacNumKb( const Eref& e ) const = 0;

		// Assignment of Km is in conc units.
		virtual void setMMenzKm( const Eref& e, double v ) const = 0;
		virtual void setMMenzKcat( const Eref& e, double v ) const = 0;
		virtual double getMMenzNumKm( const Eref& e ) const = 0;
		virtual double getMMenzKcat( const Eref& e ) const = 0;

		// Assignment of K1 is in conc units.
		virtual void setEnzK1( const Eref& e, double v ) const = 0;
		virtual void setEnzK2( const Eref& e, double v ) const = 0;
		virtual void setEnzK3( const Eref& e, double v ) const = 0;
		virtual double getEnzNumK1( const Eref& e ) const = 0;
		virtual double getEnzK2( const Eref& e ) const = 0;
		virtual double getEnzK3( const Eref& e ) const = 0;

		//////////////////////////////////////////////////////////////////
		// Junction FieldElement access functions
		//////////////////////////////////////////////////////////////////
		
		/// Returns pointer to specified Junction
		SolverJunction* getJunction( unsigned int i );
		
		/// Returns number of junctions.
		unsigned int getNumJunctions() const;

		/// Dummy function, we have to manipulate the junctions using the
		// add/drop junction functions.
		void setNumJunctions( unsigned int v );
		//////////////////////////////////////////////////////////////////
		// Cross-solver computation functions
		//////////////////////////////////////////////////////////////////
		//
		/// Handles arriving messages through junction
		void handleJunctionPoolDelta( unsigned int poolIndex, 
						vector< double > delta );
		void handleJunctionPoolNum( unsigned int poolIndex, 
						vector< double > num );

		/**
		 * addJunction:
		 * Create a junction between self and specified other SolverBase
		 * This is called only on the master solver, the one which does 
		 * the diffusion calculations. To do so, this solver expands its 
		 * own pool matrices S and Sinit to include the abutting voxels. 
		 * The follower solver sends messages to put pool #s into the 
		 * abutting voxels, and gets back changes in these pool #s. 
		 * Does not do the diffusion calculations and does not expand.
		 */
		void addJunction( const Eref& e, const Qinfo* q, Id other );
		/// Remove the junction between self and specified other SolverBase
		void dropJunction( const Eref& e, const Qinfo* q, Id other );

		/**
		 * Configures the two junctions. Used both by addJunction, and
		 * by any function that has to rebuild junctions already in place.
		 */
		void configureJunction( Id selfSolver, Id otherSolver,
					SolverJunction& selfJunc, SolverJunction& otherJunc );

		/**
		 * Scans through all junctions. If they are master junctions,
		 * reconfigures them using the configureJunction call. 
		 * Used whenever any part of the reac-diff system has been altered.
		 */
		void reconfigureAllJunctions( const Eref& e, const Qinfo* q );

		void innerConnectJunctions( 
						Id me, Id other, SolverBase* otherSP );

		/**
		 * Cleans out and reallocates solver data structs based on latest
		 * mesh and junction information. Need to call on all solvers
		 * as prelude to
		 * reconfigureAllJunctions.
		 */
		void reallocateSolver( const Eref& e, const Qinfo* q );
		virtual void innerReallocateSolver( const Eref& e ) = 0;

		virtual void expandSforDiffusion( 
			const vector< unsigned int > & otherMeshIndex,
			const vector< unsigned int > & selfDiffPoolIndex,
			SolverJunction& j ) = 0;

		void findDiffusionTerms(
				const SolverBase* otherSP,
				vector< unsigned int >& selfTerms,
				vector< unsigned int >& otherTerms
			) const;

		//////////////////////////////////////////////////////////////////
		// Matching virtual functions
		//////////////////////////////////////////////////////////////////
		/// Sends messages through junction. Called during Process.
		virtual void vUpdateJunction( const Eref& e, 
				const vector< vector< double > >& lastS,
				unsigned int threadNum, double dt ) = 0;

		/// Handles arriving messages through junction. Callsed 
		virtual void vHandleJunctionPoolDelta( unsigned int poolIndex, 
						const vector< double >& v ) = 0;
		virtual void vHandleJunctionPoolNum( unsigned int poolIndex, 
						const vector< double >& v ) = 0;

		/// Create a junction between self and specified other StoichPool
		virtual void vAddJunction( const Eref& e, const Qinfo* q, Id other ) = 0;
		/// Remove the junction between self and specified other StoichPool
		virtual void vDropJunction( const Eref& e, const Qinfo* q, Id other ) = 0;

		/**
		 * Generate the map of varPools that diffuse. 
		 */
		virtual void vBuildDiffTerms( map< string, unsigned int >& 
						diffTerms ) const = 0;

		/**
 		 * Virtual function to do any local updates to the stoich following
 		 * changes to the Junctions.
 		*/
		virtual void updateJunctionInterface( const Eref& e ) = 0;

		/**
		 * Works out which meshEntries talk to each other. The outcome
		 * is reported as 
		 * - MeshIndex vectors for self and other, to specify which entries
		 *   to use
		 * - MeshMap vectors for self and other, to map from the index
		 *   in the data transfer vector between solvers, to the meshIndex
		 *   on the solver.
		 * This function is expected to refer to a ChemCompt in order to
		 * figure out the details. Even derived classes of SolverBase
		 * that do not use a mesh (e.g., Smoldyn) will use a dummy 
		 * ChemCompt to do the needful.
		 */
		virtual void matchMeshEntries( SolverBase* other,
			vector< unsigned int >& selfMeshIndex, 
			vector< VoxelJunction >& selfMeshMap,
			vector< unsigned int >& otherMeshIndex, 
			vector< VoxelJunction >& otherMeshMap
		) const = 0;

		virtual ChemCompt* compartmentMesh() const = 0;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
		//////////////////////////////////////////////////////////////////
	private:
		/**
		 * Vector of Junctions between solvers. These specify how solvers
		 * communicate between each other in cases of diffusion, motors,
		 * or reactions. Accessed through FieldElements.
		 */
		vector< SolverJunction > junctions_;
};

/**
 * Returns Id of compartment in which specified object is located.
 * Returns Id() if no compartment found.
Id getCompt( Id id );
 */

#endif	// _SOLVER_BASE_H
