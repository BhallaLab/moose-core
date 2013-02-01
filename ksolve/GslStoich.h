/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _GSL_STOICH_H
#define _GSL_STOICH_H

class GslStoich: public SolverBase
{
	public:
		GslStoich();
		~GslStoich();
		GslStoich& operator=( const GslStoich& other );

///////////////////////////////////////////////////
// Zombie setup functions.
///////////////////////////////////////////////////
		/**
		 * Returns the master StoichCore instance which is the basis
		 * for spawning the compute Stoichs_ used in the ode_ array.
		 */
		const StoichCore* coreStoich() const;
		// Inherited reaction installation functions
		void installReaction( Id reacId, 
				const vector< Id >& subs, const vector< Id >& prds );
		void installMMenz( Id enzId, Id enzMolId, 
				const vector< Id >& subs, const vector< Id >& prds );
		void installEnzyme( Id enzId, Id enzMolId, Id cplxId,
				const vector< Id >& subs, const vector< Id >& prds );

///////////////////////////////////////////////////
// Info functions
///////////////////////////////////////////////////
		/// Returns pools_ vector
		const vector< VoxelPools >& pools();
		/// Returns ode_ vector
		const vector< OdeSystem >& ode();

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
		unsigned int getNumVarPools() const;

		/**
		 *  Returns total number of local pools. Leaves out the pools whose
		 *  actual calculations happen on another solver, but are given a
		 *  proxy here in order to handle cross-compartment reactions.
		 */
		unsigned int getNumAllPools() const;
		unsigned int getNumLocalVoxels() const;
		unsigned int getNumAllVoxels() const;
		string getPath( const Eref& e, const Qinfo* q ) const;
		/**
		 * Set up the model based on the provided wildcard path which
		 * specifies all elements managed by this solver.
		 */
		void setPath( const Eref& e, const Qinfo* q, string path );
		/**
		 * Set up the model based on the provided
		 * elist of all elements managed by this solver.
		 */
		void setElist( const Eref& e, const Qinfo* q, vector< Id > elist );
		double getEstimatedDt() const;
		bool getIsInitialized() const;
		string getMethod() const;
		void setMethod( string method );
		double getRelativeAccuracy() const;
		void setRelativeAccuracy( double value );
		double getAbsoluteAccuracy() const;
		void setAbsoluteAccuracy( double value );
		double getInternalDt() const;
		void setInternalDt( double value );
		Id getCompartment() const;
		void setCompartment( Id value );
		vector< Id > getCoupledCompartments() const;

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
		void allocatePools( unsigned int numPools );

		void process( const Eref& e, ProcPtr info );
		void reinit( const Eref& e, ProcPtr info );
		void init( const Eref& e, ProcPtr info );
		void initReinit( const Eref& e, ProcPtr info );

		void remesh( const Eref& e, const Qinfo* q,
			double oldVol,
			unsigned int numTotalEntries, unsigned int startEntry, 
			vector< unsigned int > localIndices, vector< double > vols );

///////////////////////////////////////////////////////////0
// Numerical functions
///////////////////////////////////////////////////////////0

		/// Does calculations for diffusion.
		void updateDiffusion( 
				vector< vector< double > >& lastS, 
				vector< vector< double > >& y, 
				double dt );

		/// Calculations for diffusion across junctions.
		void updateJunctionDiffusion( 
				unsigned int meshIndex, double diffScale,
				const vector< unsigned int >& diffTerms,
				double* v, double dt );

		/**
		 * Computes change in pool #s following cross-solver reacn across
		 * junction j, puts in yprime.
		 * Returns numberof terms computed.
		 */
		unsigned int fillReactionDelta( 
				const SolverJunction* j, 
				const vector< vector< double > > & lastS, 
				double* yprime ) const;

		/**
		 * Computes change in pool #s following diffusion across
		 * junction j, puts in yprime.
		 */
		void fillDiffusionDelta( 
				const SolverJunction* j, 
				const vector< vector< double > > & lastS, 
				double* yprime ) const;
		
		/**
 		 * gslFunc is the function used by GSL to advance the simulation one
 		 * step.
 		 * Perhaps not by accident, this same functional form is used by
 		 * CVODE. Should make it easier to eventually use CVODE as a 
		 * solver too.
 		 */
		static int gslFunc( double t, 
						const double* y, double* yprime, 
						void* s );
		
		/// This does the real work for GSL to advance.
		int innerGslFunc( double t, const double* y, double* yprime );
		//////////////////////////////////////////////////////////////////
		// Field access functions for pools, overriding virtual defns.
		//////////////////////////////////////////////////////////////////

		void setN( const Eref& e, double v );
		double getN( const Eref& e ) const;
		void setNinit( const Eref& e, double v );
		double getNinit( const Eref& e ) const;
		void setSpecies( const Eref& e, unsigned int v );
		unsigned int getSpecies( const Eref& e );
		void setDiffConst( const Eref& e, double v );
		double getDiffConst( const Eref& e ) const;

		//////////////////////////////////////////////////////////////////
		// Field access functions for reacs, overriding virtual defns.
		//////////////////////////////////////////////////////////////////
		// Assignment of Kf and Kb are in conc units.
		void setReacKf( const Eref& e, double v ) const;
		void setReacKb( const Eref& e, double v ) const;
		double getReacNumKf( const Eref& e ) const;
		double getReacNumKb( const Eref& e ) const;

		// Assignment of Km is in conc units.
		void setMMenzKm( const Eref& e, double v ) const;
		void setMMenzKcat( const Eref& e, double v ) const;
		double getMMenzNumKm( const Eref& e ) const;
		double getMMenzKcat( const Eref& e ) const;

		// Assignment of K1 is in conc units.
		void setEnzK1( const Eref& e, double v ) const;
		void setEnzK2( const Eref& e, double v ) const;
		void setEnzK3( const Eref& e, double v ) const;
		double getEnzNumK1( const Eref& e ) const;
		double getEnzK2( const Eref& e ) const;
		double getEnzK3( const Eref& e ) const;

		//////////////////////////////////////////////////////////////////
		// Junction operations.
		//////////////////////////////////////////////////////////////////
		/// Sends messages through junction. Called during Process.
		void vUpdateJunction( const Eref& e, 
					const vector< vector< double > >& lastS,
					unsigned int threadNum, double dt );

		/// Handles arriving messages through junction. Callsed 
		void vHandleJunctionPoolDelta( unsigned int fieldIndex, 
						const vector< double >& v );
		void vHandleJunctionPoolNum( unsigned int fieldIndex, 
						const vector< double >& v );

		/// Create a junction between self and specified other StoichPool
		void vAddJunction( const Eref& e, const Qinfo* q, Id other );
		/// Remove the junction between self and specified other StoichPool
		void vDropJunction( const Eref& e, const Qinfo* q, Id other );

		/// Find the odeSystem that matches the specified compartment sig
		unsigned int selectOde( const vector< Id >& sig ) const;

		/**
 		 * Virtual function to do any local updates to the stoich following
 		 * changes to the Junctions.
 		*/
		void updateJunctionInterface( const Eref& e );

		/// Returns indices of cross-compt reacs terms into rates_ vector.
		void vBuildReacTerms( vector< unsigned int >& reacTerms, 
				vector< pair< unsigned int, unsigned int > >& reacPoolIndex,
				Id other ) const;

		/// Returns map of diffusing pools and their names.
		void vBuildDiffTerms( map< string, unsigned int >& diffTerms ) 
				const; 

		/// Generates mapping of mesh entries between solvers.
		void matchMeshEntries( SolverBase* other,
			vector< unsigned int >& selfMeshIndex, 
			vector< VoxelJunction >& selfMeshMap,
			vector< unsigned int >& otherMeshIndex, 
			vector< VoxelJunction >& otherMeshMap
		) const;

		/// Returns pointer to ChemMesh entry for compartment.
		ChemMesh* compartmentMesh() const;

		/// Inherited virtual function, needed here to expand y_.
		void expandSforDiffusion(
			const vector< unsigned int > & otherMeshIndex,
			const vector< unsigned int > & selfDiffPoolIndex,
			SolverJunction& j );

		/// Inherited virtual func. Identifies cross-solver pools from other
		void findPoolsOnOther( Id other, vector< Id >& pools );

		/// Inherited, fills out pools involved in cross-solver reaction.
		void setLocalCrossReactingPools( const vector< Id >& pools );

		/// Inherited, returns internal solver index corresponding to poolId
		unsigned int convertIdToPoolIndex( Id id ) const;

		/// Handle calls for changing number of voxels.
		void meshSplit( 
				vector< double > initConcs,  // in milliMolar
				vector< double > vols,		// in m^3
				vector< unsigned int > localEntryList );

		const double* S( unsigned int meshIndex ) const;

		/**
		 * Generates all possible groupings of OdeSystem entries based on
		 * all possible sets of compartment signatures, starting from zero.
		 */
		unsigned int generateOdes();

///////////////////////////////////////////////////////////0
		static const Cinfo* initCinfo();
	private:
		bool isInitialized_;
		string method_;
		string path_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		vector< vector < double > >  y_;

		/**
		 * This is the master stoichCore of the system, handles all the
		 * field access and setup stuff. Typically does not do calculations
		 * itself, instead spawns compies into the ode_ vector.
		 */
		StoichCore coreStoich_;

		/**
		 * One OdeSystem for every combination of external
		 * reactions. This includes the StoichCore and the Gsl structs.
		 * The key guarantee is that the core reaction set, 
		 * that is located on the local solver, is common to all the
		 * OdeSystems. 
		 * Thus it is safe to use this core portion of pools with the 
		 * same indices for all the diffusion calculations. The
		 * specialization of each of the OdeSystems is for different sets
		 * of cross-compartment reactions, which may entail allocation of
		 * proxy pools.
		 * Note this is indexed by # of unique combinations, not by number
		 * of meshEntries. Probably much smaller than # of mesh entries.
		 */
		vector< OdeSystem > ode_;

		/// Handles set of reactant pools for each voxel.
		vector< VoxelPools > pools_;

		/**
		 * vector of indices of the local set of Voxels/MeshEntries.
		 * These may be a subset, even non-contiguous, of the entire
		 * reac-diff system, depending on how boundaries and voxelization
		 * are done.
		 * GlobalMeshIndex = localMeshEntries_[localIndex]
		 */
		vector< unsigned int > localMeshEntries_;

		Id compartmentId_;
		ChemMesh* diffusionMesh_;

		// Used to keep track of meshEntry when passing self into GSL.
		unsigned int currMeshEntry_; 
};
#endif // _GSL_STOICH_H
