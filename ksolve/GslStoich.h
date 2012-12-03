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
class GslStoich: public StoichPools
{
	public:
		GslStoich();
		~GslStoich();
		GslStoich& operator=( const GslStoich& other );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
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

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr info );
		void reinit( const Eref& e, ProcPtr info );

		void stoich( const Eref& e, const Qinfo* q, Id stoichId );

		void remesh( const Eref& e, const Qinfo* q,
			double oldVol,
			unsigned int numTotalEntries, unsigned int startEntry, 
			vector< unsigned int > localIndices, vector< double > vols );

///////////////////////////////////////////////////////////0
// Numerical functions
///////////////////////////////////////////////////////////0

		// Does calculations for diffusion.
		void updateDiffusion( 
				vector< vector< double > >& lastS, 
				vector< vector< double > >& y, 
				double dt );

		void updateJunctionDiffusion( 
				unsigned int meshIndex, double diffScale,
				const vector< unsigned int >& diffTerms,
				double* v, double dt );
		
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
		// Field access functions, overriding virtual defns.
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
		// Junction operations.
		//////////////////////////////////////////////////////////////////
		/// Sends messages through junction. Called during Process.
		void vUpdateJunction( const Eref& e, unsigned int threadNum,
					double dt );

		/// Handles arriving messages through junction. Callsed 
		void vHandleJunction( unsigned int fieldIndex, 
						const vector< double >& v );

		/// Create a junction between self and specified other StoichPool
		void vAddJunction( const Eref& e, const Qinfo* q, Id other );
		/// Remove the junction between self and specified other StoichPool
		void vDropJunction( const Eref& e, const Qinfo* q, Id other );

		/// Returns indices of cross-compt reacs terms into rates_ vector.
		void vBuildReacTerms( vector< unsigned int >& reacTerms, 
				vector< pair< unsigned int, unsigned int > >& reacPoolIndex,
				Id other ) const;

		/// Returns map of diffusing pools and their names.
		void vBuildDiffTerms( map< string, unsigned int >& diffTerms ) 
				const; 

		/// Generates mapping of mesh entries between solvers.
		void matchMeshEntries( const StoichPools* other,
			vector< unsigned int >& selfMeshIndex, 
			vector< VoxelJunction >& selfMeshMap,
			vector< unsigned int >& otherMeshIndex, 
			vector< VoxelJunction >& otherMeshMap
		) const;

		/// Returns pointer to ChemMesh entry for compartment.
		const ChemMesh* compartmentMesh() const;

///////////////////////////////////////////////////////////0
		static const Cinfo* initCinfo();
	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		vector< vector < double > >  y_;
		Id stoichId_;
		StoichCore* stoich_;

		Id compartmentId_;
		ChemMesh* diffusionMesh_;

		// Used to keep track of meshEntry when passing self into GSL.
		unsigned int currMeshEntry_; 

		// GSL stuff
		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};
#endif // _GSL_STOICH_H
