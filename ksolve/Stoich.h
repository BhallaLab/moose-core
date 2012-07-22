/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _STOICH_H
#define _STOICH_H

class Stoich
{
	public: 
		Stoich();
		~Stoich();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setOneWay( bool v );
		bool getOneWay() const;
		unsigned int getNumVarPools() const;

		void setPath( const Eref& e, const Qinfo* q, string v );
		string getPath( const Eref& e, const Qinfo* q ) const;

		unsigned int getNumMeshEntries() const;
		double getEstimatedDt() const;

		Port* getPort( unsigned int i );
		unsigned int getNumPorts() const;
		void setNumPorts( unsigned int num );

		/*
		unsigned int numCompartments() const;
		double getCompartmentVolume( short i ) const;
		void setCompartmentVolume( short comptIndex, double v );
		*/

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		/**
		 * Handles incoming messages representing influx of molecules
 		 */
		void influx( unsigned int port, vector< double > mol );

		/**
		 * Scans through incoming and self molecule list, matching up Ids
		 * to use in the port. Sets up the data structures to do so.
		 * Sends out a message indicated the selected subset.
		 */
		void handleAvailableMolsAtPort( unsigned int port, vector< SpeciesId > mols );

		/**
		 * Scans through incoming and self molecule list, checking that
		 * all match. Sets up the data structures for the port.
		 */
		void handleMatchedMolsAtPort( unsigned int port, vector< SpeciesId > mols );


		void handleRemesh( unsigned int numLocalMeshEntries, 
			vector< unsigned int > computedEntries, 
			vector< unsigned int > allocatedEntries, 
			vector< vector< unsigned int > > outgoingDiffusion, 
			vector< vector< unsigned int > > incomingDiffusion );

		void handleNodeDiffBoundary( unsigned int nodeNum, 
			vector< unsigned int > meshEntries, vector< double > remoteS );

		void meshSplit( vector< double > vols,
			vector< unsigned int > localEntryList,
			vector< vector< unsigned int > > outgoingDiffusion,
			vector< vector< unsigned int > > incomingDiffusion
		);

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		void allocateObjMap( const vector< Id >& elist );
		void allocateModel( const vector< Id >& elist );

		/**
		 * zombifyModel marches through the specified id list and 
		 * converts all entries into zombies.
		 */
		void zombifyModel( const Eref& e, const vector< Id >& elist );

		/**
		 * Converts back to EE type basic Elements.
		 */
		void unZombifyModel();
		void zombifyChemMesh( Id compt );

		unsigned int convertIdToReacIndex( Id id ) const;
		unsigned int convertIdToPoolIndex( Id id ) const;
		unsigned int convertIdToFuncIndex( Id id ) const;
		// unsigned int convertIdToComptIndex( Id id ) const;

		/**
		 * This takes the specified forward and reverse half-reacs belonging
		 * to the specified Reac, and builds them into the Stoich.
		 */
		void installReaction( ZeroOrder* forward, ZeroOrder* reverse, Id reacId );
		/**
		 * This takes the baseclass for an MMEnzyme and builds the
		 * MMenz into the Stoich.
		 */
		void installMMenz( MMEnzymeBase* meb, unsigned int rateIndex,
			const vector< Id >& subs, const vector< Id >& prds );

		/**
		 * This takes the forward, backward and product formation half-reacs
		 * belonging to the specified Enzyme, and builds them into the
		 * Stoich
		 */
		void installEnzyme( ZeroOrder* r1, ZeroOrder* r2, ZeroOrder* r3,
			Id enzId, Id enzMolId, const vector< Id >& prds );

		/**
		 * Returns the total number of entries in the mesh. This is the
		 * allocated size of the S_ matrix.
		 */
		unsigned int numMeshEntries() const;

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

		/**
		 * Utility function, used during zombification. Sets default 
		 * concInit entry: a single value for the whole pool, which applies
		 * regardless of subsequent scaling of volumes or assignment of 
		 * individual mesh values. Used only when remeshing.
		 */
		void setConcInit( unsigned int poolIndex, double conc );

		/**
		 * Returns diffusion rate of specified pool
		 */
		double getDiffConst( unsigned int poolIndex ) const;

		/**
		 * Assigns diffusion rate of specified pool
		 */
		void setDiffConst( unsigned int poolIndex, double d );

		/**
		 * Returns SpeciesId of specified pool
		 */
		SpeciesId getSpecies( unsigned int poolIndex ) const;

		/**
		 * Assigns SpeciesId of specified pool
		 */
		void setSpecies( unsigned int poolIndex, SpeciesId s );

		/**
		 * Returns working memory for the calculations at the specified
		 * mesh index
		 */
		double* getY( unsigned int meshIndex );

		/**
		 * Sets the forward rate v (given in millimoloar concentration units)
		 * for the specified reaction throughout the compartment in which the
		 * reaction lives. Internally the stoich uses #/voxel units so this 
		 * involves querying the volume subsystem about volumes for each
		 * voxel, and scaling accordingly.
		 */
		void setReacKf( const Eref& e, double v ) const;

		/**
		 * Sets the reverse rate v (given in millimoloar concentration units)
		 * for the specified reaction throughout the compartment in which the
		 * reaction lives. Internally the stoich uses #/voxel units so this 
		 * involves querying the volume subsystem about volumes for each
		 * voxel, and scaling accordingly.
		 */
		void setReacKb( const Eref& e, double v ) const;

		/**
		 * Sets the Km for MMenz, using appropriate volume conversion to
		 * go from the argument (in millimolar) to #/voxel.
		 * This may do the assignment among many voxels containing the enz
		 * in case there are different volumes.
		 */
		void setMMenzKm( const Eref& e, double v ) const;

		/**
		 * Sets the kcat for MMenz. No conversions needed.
		 */
		void setMMenzKcat( const Eref& e, double v ) const;

		/**
		 * Sets the rate v (given in millimoloar concentration units)
		 * for the forward enzyme reaction of binding substrate to enzyme.
		 * Does this throughout the compartment in which the
		 * enzyme lives. Internally the stoich uses #/voxel units so this 
		 * involves querying the volume subsystem about volumes for each
		 * voxel, and scaling accordingly.
		 */
		void setEnzK1( const Eref& e, double v ) const;

		/// Set rate k2 (1/sec) for enzyme
		void setEnzK2( const Eref& e, double v ) const;
		/// Set rate k3 (1/sec) for enzyme
		void setEnzK3( const Eref& e, double v ) const;

		/**
		 * Returns the internal rate in #/voxel, for R1, for the specified
		 * reacIndex and voxel index.
		 */
		double getR1( unsigned int reacIndex, unsigned int voxel ) const;

		/**
		 * Returns the internal rate in #/voxel, for R2, for the specified
		 * reacIndex and voxel index. In some cases R2 is undefined, and it
		 * then returns 0.
		 */
		double getR2( unsigned int reacIndex, unsigned int voxel ) const;

		/**
		 * Virtual function to assign N. Derived classes (GssaStoich) will
		 * need to do additional stuff to update dependent reacs
		 */
		virtual void innerSetN( unsigned int meshIndex, Id id, double v );

		/**
		 * Virtual function to assign Ninit. Derived classes (GssaStoich)
		 * will
		 * need to do additional stuff to update dependent reacs
		 */
		virtual void innerSetNinit( unsigned int meshIndex, Id id, double v );
		//////////////////////////////////////////////////////////////////
		// Compute functions
		//////////////////////////////////////////////////////////////////

		/**
		 * Reinitializes all variables and rates. This function may also do 
		 * reallocation, so it must be called in a thread-safe manner
		 * by whatever object directly handles the process calls.
		 */
		void innerReinit();

		/**
		 * Update the v_ vector for individual reaction velocities. Uses
		 * hooks into the S_ vector for its arguments.
		 */
		void updateV( unsigned int meshIndex, vector< double >& v );

		/**
		 * Update all the function-computed molecule terms. These are not
		 * integrated, but their values may be used by molecules that will
		 * be integrated using the solver.
		 * Uses hooks into the S_ vector for arguments other than t.
		 */
		void updateFuncs( double t, unsigned int meshIndex );

		void updateRates( vector< double>* yprime, double dt, 
			unsigned int meshIndex, vector< double >& v );

		/**
		 * Update diffusion terms for all molecules on specified meshIndex.
		 * The stencil says how to weight diffusive flux from various offset
		 * indices with respect to the current meshIndex.
		 * The first entry of the stencil is the index offset.
		 * The second entry of the stencil is the scale factor, including
		 * coeffs of that term and 1/dx^2.
		 * For example, in the Method Of Lines with second order stencil
		 * in one dimension we have:
		 * du/dt = (u_-1 - 2u + u_+1) / dx^2
		 * The scale factor for u_-1 is then 1/dx^2. Index offset is -1.
		 * The scale factor for u is then -2/dx^2. Index offset is 0.
		 * The scale factor for u_+1 is then 1/dx^2. Index offset is +1.
		 */
		void updateDiffusion( unsigned int meshIndex, 
			const vector< const Stencil* >& stencil);

		/**
		 * Clear out the flux matrix, that is the matrix of all diffusive
		 * and port-related influx and efflux from each mesh location for
		 * each molecule. This should be called after the timestep for
		 * numerical integration but before any of the flux updates
		 * (such as updateDiffusion).
		 */
		void clearFlux();
		void clearFlux( unsigned int meshIndex, unsigned int threadNum );

		/**
		 * Utility func for debugging: Prints N_ matrix
		 */
		void print() const;

#ifdef USE_GSL
		static int gslFunc( double t, const double* y, double* yprime, void* s );
		int innerGslFunc( double t, const double* y, double* yprime,
			unsigned int meshIndex );
#endif // USE_GSL


		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	protected:
		bool useOneWay_;
		string path_;
		Id stoichId_;

		/**
		 * concInit is the reference array of initial concs of molecules,
		 * in millimolar (SI units). This is non-spatial, like most of
		 * the prototype reaction systems. If the reaction is spatial
		 * these init
		 * concs will subsequently be overridden
		 */
		vector< double > concInit_;

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
		 * y_ is working memory. It maps onto S_, but stores only the 
		 * variable molecules (up to numVarPools).
		 * Has to be distinct from S because GSL uses this and swaps it
		 * back and forth with a distinct buffer.
		 * The array looks like y_[meshIndex][poolIndex]
		 * The entire y_ vector is allocated, but the pools are only 
		 * allocated for local meshEntries and for pools on
		 * diffusive boundaries with other nodes.
		 */
		vector< vector< double > > y_;

		/**
		 * Summed external flux terms for each meshpoint and each pool. 
		 * These are in units of d#/dt and add onto whatever form of 
		 * numerical integration (or stochastic calculation) is in play.
		 * Note that these terms are constant for the entire duration of
		 * one clock tick, so it represents a first order Euler integration.
		 * The clock tick has to be set with this recognized.
		 */
		vector< vector< double > > flux_;

		/**
		 * vector of indices of meshEntries to be computed locally.
		 * This may not necessarily be a contiguous set, depending on
		 * how boundaries and voxelization is done.
		 */
		vector< unsigned int > localMeshEntries_;

		/**
		 * List of target nodes
		 */
		vector< unsigned int > diffNodes_;

		/**
		 * outgoing_[targetNode][meshEntries]
		 * For each target node, this provides a list of meshEntries to
		 * transmit. Note that for each meshEntry the entire list of 
		 * diffusive molecules is sent across.
		 */
		vector< vector< unsigned int > > outgoing_;

		/**
		 * incoming_[targetNode][meshEntries]
		 * For each target node, this provides a list of meshEntries that
		 * are received and put into the appropriate locations on the 
		 * S_ vector.
		 */
		vector< vector< unsigned int > > incoming_;

		/**
		 * Vector of diffusion constants, one per VarPool.
		 */
		vector< double > diffConst_;

		/**
		 * Vector of indices for non-zero diffusion constants. Later.
		vector< unsigned int > indexOfDiffusingPools_;
		 */

		/**
		 * Lookup from each molecule to its parent compartment index
		 * compartment_.size() == number of distinct pools == max poolIndex
		vector< short > compartment_;
		 */

		/**
		 * Lookup from each molecule to its Species identifer
		 * This will eventually be tied into an ontology reference.
		 */
		vector< SpeciesId > species_;

		/// The RateTerms handle the update operations for reaction rate v_
		vector< RateTerm* > rates_;

		/// The FuncTerms handle mathematical ops on mol levels.
		vector< FuncTerm* > funcs_;

		/// N_ is the stoichiometry matrix.
		KinSparseMatrix N_;


		/**
		 * totPortSize_: The sum of all port entries
		 */
		unsigned int totPortSize_;

		/**
		 * Maps Ids to objects in the S_, RateTerm, and FuncTerm vectors.
		 * There will be holes in this map, but look up is very fast.
		 * The calling Id must know what it wants to find: all it
		 * gets back is an integer.
		 * The alternative is to have multiple maps, but that is slower.
		 * Assume no arrays. Each Pool/reac etc must be a unique
		 * Element. Later we'll deal with diffusion.
		 */
		vector< unsigned int > objMap_;
		/**
		 * Minor efficiency: We will usually have a set of objects that are
		 * nearly contiguous in the map. May as well start with the first of
		 * them.
		 */
		unsigned int objMapStart_;

		/**
		 * Map back from mol index to Id. Primarily for debugging.
		 */
		vector< Id > idMap_;
		
		/**
		 * Number of variable molecules that the solver deals with.
		 *
		 */
		unsigned int numVarPools_;
		unsigned int numVarPoolsBytes_;
		/**
		 * Number of buffered molecules
		 */
		unsigned int numBufPools_;
		/**
		 * Number of molecules whose values are computed by functions
		 */
		unsigned int numFuncPools_;

		/**
		 * Number of reactions in the solver model. This includes 
		 * conversion reactions A + B <---> C
		 * enzyme reactions E + S <---> E.S ---> E + P
		 * and MM enzyme reactions rate = E.S.kcat / ( Km + S )
		 * The enzyme reactions count as two reaction steps.
		 */
		unsigned int numReac_;

		/**
		 * The Ports are interfaces to other solvers by way of a spatial
		 * junction between the solver domains. They manage 
		 * the info about which molecules exchange, 
		 * They are also the connection point for the messages that 
		 * handle the port data transfer.
		 * Each Port connects to exactly one other solver.
		 */
		vector< Port > ports_;
};

class StoichThread
{
	public:
		StoichThread()
			: s_( 0 ), p_( 0 ), meshIndex_( 0 )
		{;}

		void set( Stoich* s, const ProcInfo* p, unsigned int m )
		{
			s_ = s;
			p_ = p;
			meshIndex_ = m;
		}

		Stoich* stoich() const {
			return s_;
		}

		const ProcInfo* procInfo() const {
			return p_;
		}

		unsigned int meshIndex() const {
			return meshIndex_;
		}
	
	private:
		Stoich* s_;
		const ProcInfo* p_;
		unsigned int meshIndex_;
};

#endif	// _STOICH_H
