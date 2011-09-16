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

		Port* getPort( unsigned int i );
		unsigned int getNumPorts() const;
		void setNumPorts( unsigned int num );

		unsigned int numCompartments() const;
		double getCompartmentVolume( short i ) const;
		void setCompartmentVolume( short comptIndex, double v );

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		/**
		 * Handles incoming messages representing influx of molecules
 		 */
		void influx( DataId port, vector< double > mol );

		/**
		 * Scans through incoming and self molecule list, matching up Ids
		 * to use in the port. Sets up the data structures to do so.
		 * Sends out a message indicated the selected subset.
		 */
		void handleAvailableMolsAtPort( DataId port, vector< SpeciesId > mols );

		/**
		 * Scans through incoming and self molecule list, checking that
		 * all match. Sets up the data structures for the port.
		 */
		void handleMatchedMolsAtPort( DataId port, vector< SpeciesId > mols );

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		void allocateObjMap( const vector< Id >& elist );
		void allocateModel( const vector< Id >& elist );
		void zombifyModel( const Eref& e, const vector< Id >& elist );
		void zombifyChemMesh( Id compt );

		unsigned int convertIdToReacIndex( Id id ) const;
		unsigned int convertIdToPoolIndex( Id id ) const;
		unsigned int convertIdToFuncIndex( Id id ) const;
		unsigned int convertIdToComptIndex( Id id ) const;

		const double* S() const;
		double* varS();
		const double* Sinit() const;
		double* getY();

		//////////////////////////////////////////////////////////////////
		// Compute functions
		//////////////////////////////////////////////////////////////////
		/**
		 * Update the v_ vector for individual reaction velocities. Uses
		 * hooks into the S_ vector for its arguments.
		 */
		void updateV( );

		/**
		 * Update all the function-computed molecule terms. These are not
		 * integrated, but their values may be used by molecules that will
		 * be integrated using the solver.
		 * Uses hooks into the S_ vector for arguments other than t.
		 */
		void updateFuncs( double t );

		void updateRates( vector< double>* yprime, double dt  );

#ifdef USE_GSL
		static int gslFunc( double t, const double* y, double* yprime, void* s );
		int innerGslFunc( double t, const double* y, double* yprime );
#endif // USE_GSL


		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	protected:
		bool useOneWay_;
		string path_;

		/**
		 * This is the array of molecules. Of these, the first numVarPools_
		 * are variables and are integrated using the ODE solver. 
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
		vector< double > Sinit_;

		/**
		 * Lookup from each molecule to its parent compartment index
		 */
		vector< short > compartment_;

		/**
		 * Lookup from each molecule to its Species identifer
		 * This will eventually be tied into an ontology reference.
		 */
		vector< SpeciesId > species_;

		/**
		 * Size of each compartment
		 */
		vector< double > compartmentSize_;

		/**
		 * Number of subdivisions of compartment. Actually should be
		 * dimensions.
		 */
		vector< short > compartmentVoxels_;

		/// v_ holds the rates of each reaction
		vector< double > v_;

		/// The RateTerms handle the update operations for reaction rate v_
		vector< RateTerm* > rates_;

		/// The FuncTerms handle mathematical ops on mol levels.
		vector< FuncTerm* > funcs_;

		/// N_ is the stoichiometry matrix.
		KinSparseMatrix N_;

		/**
		 * y_ is working memory, only the variable molecule levels. 
		 * Should be possible to replace with S.
		 */
		vector< double > y_;

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

#endif	// _STOICH_H
