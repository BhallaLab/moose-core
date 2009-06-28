/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This program works out a steady-state value for a reaction system.
 * It uses GSL heavily, and isn't even compiled if the flag isn't set.
 * It finds the ss value closest to the initial conditions.
 *
 * If you want to find multiple stable states, use the MultiStable object,
 * which operates a SteadyState object to find multiple states.
 * If you want to carry out a dose-response calculation, use the 
 * DoseResponse object.
 * If you want to follow a stable state in phase space, use the closely
 * related StateTrajectory object.
 */

#include "moose.h"
#include "RateTerm.h"
#include "KinSparseMatrix.h"
#include "InterSolverFlux.h"
#include "Stoich.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>

#include "SteadyState.h"

int ss_func( const gsl_vector* x, void* params, gsl_vector* f );
int convert_to_row_echelon( gsl_matrix* LU, gsl_permutation* P1, 
	int num_rows, int num_columns );

// Limit below which small numbers are treated as zero.
const double SteadyState::EPSILON = 1e-9;
/**
 * This is used by the multidimensional root finder
 */
struct reac_info
{
	int rank;
	int num_reacs;
	int num_mols;
	int nIter;
	double convergenceCriterion;

	double* T;
	Stoich* s;

	gsl_matrix* Nr;
	gsl_matrix* gamma;
};

const Cinfo* initSteadyStateCinfo()
{
	/**
	 * This picks up the entire Stoich data structure
	 */
	static Finfo* gslShared[] =
	{
		new SrcFinfo( "reinitSrc", Ftype0::global() ),
		new DestFinfo( "assignStoich",
			Ftype1< void* >::global(),
			RFCAST( &SteadyState::assignStoichFunc )
			),
		new DestFinfo( "setMolN",
			Ftype2< double, unsigned int >::global(),
			RFCAST( &SteadyState::setMolN )
			),
	};

	/**
	 * These are the fields of the SteadyState class
	 */
	static Finfo* steadyStateFinfos[] =
	{
		///////////////////////////////////////////////////////
		// Field definitions
		///////////////////////////////////////////////////////
		new ValueFinfo( "badStoichiometry", 
			ValueFtype1< bool >::global(),
			GFCAST( &SteadyState::badStoichiometry ), 
			&dummyFunc,
			"True if the model has an illegal stoichiometry"
		),
		new ValueFinfo( "isInitialized", 
			ValueFtype1< bool>::global(),
			GFCAST( &SteadyState::isInitialized ), 
			&dummyFunc,
			"True if the model has been initialized successfully"
		),
		new ValueFinfo( "nIter", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getNiter ), 
			&dummyFunc
		),
		new ValueFinfo( "status", 
			ValueFtype1< string >::global(),
			GFCAST( &SteadyState::getStatus ), 
			&dummyFunc
		),
		new ValueFinfo( "maxIter", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getMaxIter ), 
			RFCAST( &SteadyState::setMaxIter )
		),
		new ValueFinfo( "convergenceCriterion", 
			ValueFtype1< double >::global(),
			GFCAST( &SteadyState::getConvergenceCriterion ), 
			RFCAST( &SteadyState::setConvergenceCriterion )
		),
		new ValueFinfo( "rank", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getRank ), 
			&dummyFunc
		),
		new LookupFinfo( "total",
			LookupFtype< double, unsigned int >::global(),
				GFCAST( &SteadyState::getTotal ),
				RFCAST( &SteadyState::setTotal ),
				"Totals table for conservation laws. The exact mapping of"
				"this to various sums of molecules is given by the "
				"conservation matrix, and is currently a bit opaque."
				"The value of 'total' is set to initial conditions when"
				"the 'SteadyState::settle' function is called."
				"Assigning values to the total is a special operation:"
				"it rescales the concentrations of all the affected"
				"molecules so that they are at the specified total."
				"This happens the next time 'settle' is called."
		),
		///////////////////////////////////////////////////////
		// MsgSrc definitions
		///////////////////////////////////////////////////////
		
		///////////////////////////////////////////////////////
		// MsgDest definitions
		///////////////////////////////////////////////////////
		new DestFinfo( "setupMatrix", 
			Ftype0::global(),
			RFCAST( &SteadyState::setupMatrix ),
			"This function initializes and rebuilds the matrices used in the calculation. It is called automatically by the KineticManager. Users will not usually have to call this."
		),
		new DestFinfo( "settle", 
			Ftype0::global(),
			RFCAST( &SteadyState::settleFunc ),
			"Finds the nearest steady state to the current initial conditions. This function rebuilds the entire calculation only if the object has not yet been initialized."
		),
		new DestFinfo( "resettle", 
			Ftype0::global(),
			RFCAST( &SteadyState::resettleFunc ),
			"Finds the nearest steady state to the current initial conditions. This function forces a rebuild of the entire calculation"
		),
		
		///////////////////////////////////////////////////////
		// Shared definitions
		///////////////////////////////////////////////////////
		new SharedFinfo( "gsl", gslShared, 
				sizeof( gslShared )/ sizeof( Finfo* ),
					"Messages that connect to the GslIntegrator object" ),

	};
	
	static string doc[] =
	{
		"Name", "SteadyState",
		"Author", "Upinder S. Bhalla, 2009, NCBS",
		"Description", "SteadyState: works out a steady-state value for "
		"a reaction system. It uses GSL heavily, and isn't even compiled "
		"if the flag isn't set. It finds the ss value closest to the "
		"initial conditions, defined by current molecular concentrations."
 "If you want to find multiple stable states, use the MultiStable object,"
 "which operates a SteadyState object to find multiple states."
	"If you want to carry out a dose-response calculation, use the "
 	"DoseResponse object."
 	"If you want to follow a stable state in phase space, use the "
	"StateTrajectory object. "
	};
	
	static Cinfo steadyStateCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		steadyStateFinfos,
		sizeof( steadyStateFinfos )/sizeof(Finfo *),
		ValueFtype1< SteadyState >::global(),
		0,
		0
	);

	return &steadyStateCinfo;
}

static const Cinfo* steadyStateCinfo = initSteadyStateCinfo();

static const Slot reinitSlot =
	initSteadyStateCinfo()->getSlot( "gsl.reinitSrc" );

///////////////////////////////////////////////////
// Class function definitions
///////////////////////////////////////////////////

SteadyState::SteadyState()
	:
		nIter_( 0 ), 
		maxIter_( 100 ), 
		badStoichiometry_( 0 ),
		status_( "OK" ),
		isInitialized_( 0 ),
		isSetup_( 0 ),
		convergenceCriterion_( 1e-7 ),
		LU_( 0 ),
		Nr_( 0 ),
		gamma_( 0 ),
		s_( 0 ),
		nVarMols_( 0 ),
		nReacs_( 0 ),
		rank_( 0 ),
		reassignTotal_( 0 )
{
	;
}

SteadyState::~SteadyState()
{
	if ( LU_ != 0 )
		gsl_matrix_free( LU_ );
	if ( Nr_ != 0 )
		gsl_matrix_free( Nr_ );
	if ( gamma_ != 0 )
		gsl_matrix_free( gamma_ );
}
		
///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

bool SteadyState::badStoichiometry( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->badStoichiometry_;
}

bool SteadyState::isInitialized( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->isInitialized_;
}

unsigned int SteadyState::getNiter( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->nIter_;
}

string SteadyState::getStatus( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->status_;
}

unsigned int SteadyState::getMaxIter( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->maxIter_;
}

void SteadyState::setMaxIter( const Conn* c, unsigned int value ) {
	static_cast< SteadyState* >( c->data() )->maxIter_ = value;
}

unsigned int SteadyState::getRank( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->rank_;
}

void SteadyState::setConvergenceCriterion( const Conn* c, double value ) {
	if ( value > 1e-10 )
		static_cast< SteadyState* >( c->data() )->convergenceCriterion_ =
			value;
	else
		cout << "Warning: Convergence criterion " << value << 
		" too small. Old value " << 
		static_cast< SteadyState* >( c->data() )->convergenceCriterion_ <<
		" retained\n";
}

double SteadyState::getConvergenceCriterion( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->
		convergenceCriterion_;
}

double SteadyState::getTotal( Eref e, const unsigned int& i )
{
	return static_cast< const SteadyState* >( e.data() )->localGetTotal(i);
}

void SteadyState::setTotal( 
	const Conn* c, double val, const unsigned int& i )
{
	static_cast< SteadyState* >( c->data() )->localSetTotal(val, i);
}

double SteadyState::localGetTotal( const unsigned int& i ) const
{
	if ( i < total_.size() )
		return total_[i];
	cout << "Warning: SteadyState::localGetTotal: index " << i <<
			" out of range " << total_.size() << endl;
	return 0.0;
}

void SteadyState::localSetTotal( double val, const unsigned int& i )
{
	if ( i < total_.size() ) {
		total_[i] = val;
		reassignTotal_ = 1;
		return;
	}
	cout << "Warning: SteadyState::localSetTotal: index " << i <<
		" out of range " << total_.size() << endl;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

// Static func
void SteadyState::setupMatrix( const Conn* c )
{
	static_cast< SteadyState* >( c->data() )->setupSSmatrix();
}
void SteadyState::settleFunc( const Conn* c )
{
	static_cast< SteadyState* >( c->data() )->settle( 0 );
}

void SteadyState::resettleFunc( const Conn* c )
{
	static_cast< SteadyState* >( c->data() )->settle( 1 );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////

void SteadyState::assignStoichFunc( const Conn* c, void* s )
{
	static_cast< SteadyState* >( c->data() )->assignStoichFuncLocal( s );
}

void SteadyState::setMolN( const Conn* c, double y, unsigned int i )
{
}

///////////////////////////////////////////////////
// GSL interface stuff
///////////////////////////////////////////////////

/**
 * This function should also set up the sizes, and it should be at 
 * allocate, not reinit time.
 */
void SteadyState::assignStoichFuncLocal( void* stoich ) 
{
	s_ = static_cast< Stoich* >( stoich );
	nVarMols_ = s_->nVarMols();
	nReacs_ = s_->velocity().size();
	isInitialized_ = 1;
	if ( !isSetup_ )
		setupSSmatrix();
}

void print_gsl_mat( gsl_matrix* m, const char* name )
{
    size_t i, j;
    printf( "%s[%ld, %ld] = \n", name, m->size1, m->size2 );
    for (i = 0; i < m->size1; i++) {
        for (j = 0; j < m->size2; j++) {
            double x = gsl_matrix_get (m, i, j );
            if ( fabs( x ) < 1e-9 ) x = 0;
            printf( "%6g", x );
        }
    
        printf( "\n");
    }   
}

void SteadyState::setupSSmatrix()
{
	if ( nVarMols_ == 0 || nReacs_ == 0 )
		return;
	
	int nTot = nVarMols_ + nReacs_;
	gsl_matrix* N = gsl_matrix_calloc (nVarMols_, nReacs_);
	gsl_permutation* P = gsl_permutation_calloc( nTot );
	int signum = 0;
	if ( LU_ ) { // Clear out old one.
		gsl_matrix_free( LU_ );
	}
	LU_ = gsl_matrix_calloc (nTot, nTot);

	for ( unsigned int i = 0; i < nVarMols_; ++i ) {
		gsl_matrix_set (LU_, i, i + nReacs_, 1 );
		for ( unsigned int j = 0; j < nReacs_; ++j ) {
			double x = s_->getStoichEntry(i, j);
			gsl_matrix_set (N, i, j, x);
			gsl_matrix_set (LU_, i, j, x );
		}
	}

	gsl_linalg_LU_decomp( LU_, P, &signum );

	rank_ = convert_to_row_echelon( LU_, P, nVarMols_, nReacs_ );
	unsigned int nConsv = nVarMols_ - rank_;
	
	if ( Nr_ ) { // Clear out old one.
		gsl_matrix_free( Nr_ );
	}
	Nr_ = gsl_matrix_calloc ( rank_, nReacs_ );
	// Fill up Nr.
	for ( unsigned int i = 0; i < rank_; i++)
		for ( unsigned int j = i; j < nReacs_; j++)
			gsl_matrix_set (Nr_, i, j, gsl_matrix_get( LU_, i, j ) );

	if ( gamma_ ) { // Clear out old one.
		gsl_matrix_free( gamma_ );
	}
	gamma_ = gsl_matrix_calloc (nConsv, nVarMols_ );
	
	// Fill up gamma
	for ( unsigned int i = rank_; i < nVarMols_; ++i )
		for ( unsigned int j = 0; j < nVarMols_; ++j )
			gsl_matrix_set( gamma_, i - rank_, j, 
				gsl_matrix_get( LU_, i, j + nReacs_ ) );

	// Fill up boundary condition values
	total_.resize( nConsv );
	total_.assign( nConsv, 0.0 );
	for ( unsigned int i = 0; i < nConsv; ++i )
		for ( unsigned int j = 0; j < nVarMols_; ++j )
			total_[i] += gsl_matrix_get( gamma_, i, j ) * s_->Sinit()[ j ];

	gsl_matrix_free( N );
	gsl_permutation_free( P );

	isSetup_ = 1;
}

static double op( double x )
{
	return x * x;
}

static double invop( double x )
{
	if ( x > 0.0 )
		return sqrt( x );
	return 0.0;
}


/**
 * This does the iteration, using the specified method.
 * First try gsl_multiroot_fsolver_hybrids
 * If that doesn't work try gsl_multiroot_fsolver_dnewton
 * Returns the gsl status.
struct reac_info
{
	int rank;
	int num_reacs;
	int num_mols;

	double* T;
	Stoich* s;

	gsl_matrix* Nr;
	gsl_matrix* gamma;
};
 */
int iterate( const gsl_multiroot_fsolver_type* st, struct reac_info *ri,
	int maxIter )
{
	gsl_vector* x = gsl_vector_calloc( ri->num_mols );
	gsl_multiroot_fsolver *solver = 
		gsl_multiroot_fsolver_alloc( st, ri->num_mols );
	gsl_multiroot_function func = {&ss_func, ri->num_mols, ri};

	// This gives the starting point for finding the solution
	for ( int i = 0; i < ri->num_mols; ++i )
		gsl_vector_set( x, i, invop( ri->s->S()[i] ) );

	gsl_multiroot_fsolver_set( solver, &func, x );

	ri->nIter = 0;
	int status;
	do {
		ri->nIter++;
		status = gsl_multiroot_fsolver_iterate( solver );
		if (status ) break;
		status = gsl_multiroot_test_residual( 
			solver->f, ri->convergenceCriterion);
	} while (status == GSL_CONTINUE && ri->nIter < maxIter );

	gsl_multiroot_fsolver_free( solver );
	gsl_vector_free( x );
	return status;
}

/**
 * The settle function computes the steady state nearest the initial
 * conditions.
 */
void SteadyState::settle( bool forceSetup )
{
	if ( !isInitialized_ ) {
		cout << "Error: SteadyState object has not been initialized. No calculations done\n";
		return;
	}
	if ( forceSetup || isSetup_ == 0 ) {
		setupSSmatrix();
	}

	// Setting up matrices and vectors for the calculation.
	unsigned int nConsv = nVarMols_ - rank_;
//	gsl_matrix* Nr = gsl_matrix_calloc ( rank_, nReacs_ );
//	gsl_matrix* gamma = gsl_matrix_calloc ( nConsv, nVarMols_ );
	double * T = (double *) calloc( nConsv, sizeof( double ) );

	unsigned int i, j;

/*
	// Fill up Nr.
	for ( i = 0; i < rank_; i++)
		for (j = i; j < nReacs_; j++)
			gsl_matrix_set (Nr, i, j, gsl_matrix_get( LU_, i, j ) );
	
	// Fill up gamma
	for ( i = rank_; i < nVarMols_; ++i )
		for ( j = 0; j < nVarMols_; ++j )
			gsl_matrix_set( gamma, i - rank_, j, 
				gsl_matrix_get( LU_, i, j + nReacs_ ) );
*/
	// Fill up boundary condition values
	if ( reassignTotal_ ) { // The user has defined new conservation values.
		for ( i = 0; i < nConsv; ++i )
			T[i] = total_[i];
		reassignTotal_ = 0;
	} else {
		for ( i = 0; i < nConsv; ++i )
			for ( j = 0; j < nVarMols_; ++j )
				T[i] += gsl_matrix_get( gamma_, i, j ) * s_->S()[ j ];
		total_.assign( T, T + nConsv );
	}

	struct reac_info ri;
	ri.rank = rank_;
	ri.num_reacs = nReacs_;
	ri.num_mols = nVarMols_;
	ri.T = T;
	ri.Nr = Nr_;
	ri.gamma = gamma_;
	ri.s = s_;
	ri.convergenceCriterion = convergenceCriterion_;

	int status = iterate( gsl_multiroot_fsolver_hybrids, &ri, maxIter_ );
	if ( status ) // It failed. Fall back with the Newton method
		status = iterate( gsl_multiroot_fsolver_dnewton, &ri, maxIter_ );
	status_ = string( gsl_strerror( status ) );
	nIter_ = ri.nIter;
	if ( status == GSL_SUCCESS ) {
		
		/*
		 * Should happen in the ss_func.
		for ( i = 0; i < nVarMols_; ++i )
			s_->S()[i] = gsl_vector_get( op( solver->x ), i );
			*/
	} else {
		cout << "Warning: SteadyState iteration failed, status = " <<
			status_ << ", nIter = " << nIter_ << endl;
	}

	// Clean up.
	// gsl_matrix_free( Nr );
	// gsl_matrix_free( gamma );
	free( T );
}

int ss_func( const gsl_vector* x, void* params, gsl_vector* f )
{
	struct reac_info* ri = (struct reac_info *)params;
	int num_consv = ri->num_mols - ri->rank;
	Stoich* s = ri->s;

	for ( int i = 0; i < ri->num_mols; ++i )
		s->S()[i] = op( gsl_vector_get( x, i ) );
	s->updateV();

	// y = Nr . v
	// Note that Nr is row-echelon: diagonal and above.
	for ( int i = 0; i < ri->rank; ++i ) {
		double temp = 0;
		for ( int j = i; j < ri->num_reacs; ++j )
			temp += gsl_matrix_get( ri->Nr, i, j ) * s->velocity()[j];
		gsl_vector_set( f, i, temp );
	}

	// dT = gamma.S - T
	for ( int i = 0; i < num_consv; ++i ) {
		double dT = - ri->T[i];
		for ( int j = 0; j < ri->num_mols; ++j )
			dT += gsl_matrix_get( ri->gamma, i, j) * 
				op( gsl_vector_get( x, j ) );

		gsl_vector_set( f, i + ri->rank, dT );
	}

	return GSL_SUCCESS;
}

int findStart( gsl_matrix* LU, int start )
{
	size_t i;
	for ( i = start; i < LU->size2; ++i )
		if ( fabs( gsl_matrix_get( LU, start, i ) ) > SteadyState::EPSILON )
			return i;
	return i;
}

// Cleans up LU matrix upper right corner, the part that specifies the
// N matrix. Converts to row echelon form, where the upper triangle are
// stoichiometries of the independent reactions. The remainder are zeros,
// and are dependent reactions.
// Returns rank.
int convert_to_row_echelon( gsl_matrix* LU, gsl_permutation* P1, 
	int num_rows, int num_columns )
{
	unsigned int width = num_rows;
	if ( num_rows > num_columns )
		width = num_columns;

	gsl_matrix* LU2 = gsl_matrix_calloc( LU->size1, LU->size2 );
	gsl_matrix_memcpy( LU2, LU );
	gsl_vector* startColumn = gsl_vector_calloc( P1->size );
	gsl_permutation* P2 = gsl_permutation_calloc( P1->size );
	gsl_permutation* P3 = gsl_permutation_calloc( P1->size );
	// int *startColumn = (int *) calloc( width, sizeof( int ) );
	size_t i;
	int rank = 0;
	for (i = 0; i < P1->size; i++) {
		unsigned int start = findStart( LU, i );
		gsl_vector_set( startColumn, i, start );
		if ( start < width )
			rank++;
	}
	gsl_sort_vector_index( P2, startColumn );

	// Apply P2 to LU.
	gsl_vector* temp = gsl_vector_calloc( LU->size2 );
	for ( i = 0; i < width; ++i ) {
		gsl_matrix_get_row (temp, LU2, i );
		gsl_matrix_set_row( LU, gsl_permutation_get( P2, i ), temp );
	}
	
	// Generate the final permutation, will need to return to apply to
	// the molecule vector.
	gsl_permutation_mul( P3, P2, P1 );

	// Assign P3 to P1.
	gsl_permutation_memcpy( P1, P3 );

	return rank;
}


