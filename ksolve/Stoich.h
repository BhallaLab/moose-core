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
		double getNumVarMols() const;

		void setPath( const Eref& e, const Qinfo* q, string v );
		string getPath( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		void allocateObjMap( const vector< Id >& elist );
		void allocateModel( const vector< Id >& elist );
		void zombifyModel( const Eref& e, const vector< Id >& elist );
		unsigned int convertIdToReacIndex( Id id ) const;
		unsigned int convertIdToMolIndex( Id id ) const;
		unsigned int convertIdToFuncIndex( Id id ) const;

		const double* S() const;
		double* varS();
		const double* Sinit() const;

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
		 * This is the array of molecules. Of these, the first numVarMols_
		 * are variables and are integrated using the ODE solver. 
		 * The next numBufMols_ are fixed but can be changed by the script.
		 * The next numFuncMols_ are computed using arbitrary functions of
		 * any of the molecule levels, and the time.
		 * The functions evaluate _before_ the ODE. 
		 * The functions should not cascade as there is no guarantee of
		 * execution order.
		 */
		vector< double > S_;
		vector< double > Sinit_;
		vector< double > v_;
		vector< RateTerm* > rates_;
		vector< FuncTerm* > funcs_;
		KinSparseMatrix N_;

		/**
		 * Maps Ids to objects in the S_, RateTerm, and FuncTerm vectors.
		 * There will be holes in this map, but look up is very fast.
		 * The calling Id must know what it wants to find: all it
		 * gets back is an integer.
		 * The alternative is to have multiple maps, but that is slower.
		 * Assume no arrays. Each Mol/reac etc must be a unique
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
		unsigned int numVarMols_;
		unsigned int numVarMolsBytes_;
		/**
		 * Number of buffered molecules
		 */
		unsigned int numBufMols_;
		/**
		 * Number of molecules whose values are computed by functions
		 */
		unsigned int numFuncMols_;

		/**
		 * Number of reactions in the solver model. This includes 
		 * conversion reactions A + B <---> C
		 * enzyme reactions E + S <---> E.S ---> E + P
		 * and MM enzyme reactions rate = E.S.kcat / ( Km + S )
		 * The enzyme reactions count as two reaction steps.
		 */
		unsigned int numReac_;
};

#endif	// _STOICH_H
