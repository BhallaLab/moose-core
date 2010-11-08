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
#include <gsl/gsl_eigen.h>

#include "SteadyState.h"

int ss_func( const gsl_vector* x, void* params, gsl_vector* f );
int myGaussianDecomp( gsl_matrix* U );

// Limit below which small numbers are treated as zero.
const double SteadyState::EPSILON = 1e-9;

// This fraction of molecules is used as an increment in computing the
// Jacobian
const double SteadyState::DELTA = 1e-6;
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
		new SrcFinfo( "requestYsrc", Ftype0::global() ),
		new DestFinfo( "assignY",
			Ftype1< double* >::global(),
			RFCAST( &SteadyState::assignY )
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
		new ValueFinfo( "nVarMols", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getNvarMols ), 
			&dummyFunc,
			"Number of variable molecules in reaction system."
		),
		new ValueFinfo( "rank", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getRank ), 
			&dummyFunc,
			"Number of independent molecules in reaction system"
		),
		new ValueFinfo( "stateType", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getStateType ), 
			&dummyFunc,
			"0: stable; 1: unstable; 2: saddle; 3: osc?; 4: one near-zero eigenvalue; 5: other"
		),
		new ValueFinfo( "nNegEigenvalues", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getNnegEigenvalues ), 
			&dummyFunc
		),
		new ValueFinfo( "nPosEigenvalues", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getNposEigenvalues ), 
			&dummyFunc
		),
		new ValueFinfo( "solutionStatus", 
			ValueFtype1< unsigned int >::global(),
			GFCAST( &SteadyState::getSolutionStatus ), 
			&dummyFunc,
			"0: Good; 1: Failed to find steady states; 2: Failed to find eigenvalues"
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
		new LookupFinfo( "eigenvalues",
			LookupFtype< double, unsigned int >::global(),
				GFCAST( &SteadyState::getEigenvalue ),
				RFCAST( &SteadyState::setEigenvalue ),
				"Eigenvalues computed for steady state"
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
		new DestFinfo( "showMatrices", 
			Ftype0::global(),
			RFCAST( &SteadyState::showMatricesFunc ),
			"Utility function to show the matrices derived for the calculations on the reaction system. Shows the Nr, gamma, and total matrices"
		),
		new DestFinfo( "randomInit", 
			Ftype0::global(),
			RFCAST( &SteadyState::randomizeInitialConditionFunc ),
			"Generate random initial conditions consistent with the mass"
			"conservation rules. Typically invoked by the StateScanner"
			"object, which will then go through the process of coordinating"
			"settle time and the SteadyState object to find"
			"the associated steady state."
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
static const Slot requestYslot =
	initSteadyStateCinfo()->getSlot( "gsl.requestYsrc" );

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
		reassignTotal_( 0 ),
		nNegEigenvalues_( 0 ),
		nPosEigenvalues_( 0 ),
		stateType_( 0 ),
		solutionStatus_( 0 ),
		numFailed_( 0 )
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

unsigned int SteadyState::getNvarMols( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->nVarMols_;
}

unsigned int SteadyState::getStateType( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->stateType_;
}

unsigned int SteadyState::getNnegEigenvalues( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->nNegEigenvalues_;
}

unsigned int SteadyState::getNposEigenvalues( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->nPosEigenvalues_;
}

unsigned int SteadyState::getSolutionStatus( Eref e ) {
	return static_cast< const SteadyState* >( e.data() )->solutionStatus_;
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


double SteadyState::getEigenvalue( Eref e, const unsigned int& i )
{
	return static_cast< const SteadyState* >( e.data() )->localGetEigenvalue(i);
}

void SteadyState::setEigenvalue( 
	const Conn* c, double val, const unsigned int& i )
{
	cout << "Warning: SteadyState::localSetEigenvalue: Readonly\n";
}

double SteadyState::localGetEigenvalue( const unsigned int& i ) const
{
	if ( i < eigenvalues_.size() )
		return eigenvalues_[i];
	cout << "Warning: SteadyState::localGetEigenvalue: index " << i <<
			" out of range " << eigenvalues_.size() << endl;
	return 0.0;
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

void SteadyState::showMatricesFunc( const Conn* c )
{
	static_cast< SteadyState* >( c->data() )->showMatrices();
}

/**
 * Initializes the system to a random initial condition that is
 * consistent with the conservation laws.
 */
void SteadyState::randomizeInitialConditionFunc( const Conn* c )
{
	static_cast< SteadyState* >( c->data() )->
		randomizeInitialCondition( c->target() );
}

// Dummy function
void SteadyState::assignY( const Conn* c, double* S )
{
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
	// This is a bad time to setup the SS matrix, since the
	// totals may change later. Have to update totals when needed.
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

void SteadyState::showMatrices()
{
	if ( !isInitialized_ ) {
		cout << "SteadyState::showMatrices: Sorry, the system is not yet initialized.\n";
		return;
	}
	int numConsv = nVarMols_ - rank_;
	cout << "Totals:	";
	for ( int i = 0; i < numConsv; ++i )
		cout << total_[i] << "	";
	cout << endl;
	print_gsl_mat( gamma_, "gamma" );
	print_gsl_mat( Nr_, "Nr" );
	print_gsl_mat( LU_, "LU" );
}

void SteadyState::setupSSmatrix()
{
	if ( nVarMols_ == 0 || nReacs_ == 0 )
		return;
	
	int nTot = nVarMols_ + nReacs_;
	gsl_matrix* N = gsl_matrix_calloc (nVarMols_, nReacs_);
	if ( LU_ ) { // Clear out old one.
		gsl_matrix_free( LU_ );
	}
	LU_ = gsl_matrix_calloc (nVarMols_, nTot);

	for ( unsigned int i = 0; i < nVarMols_; ++i ) {
		gsl_matrix_set (LU_, i, i + nReacs_, 1 );
		for ( unsigned int j = 0; j < nReacs_; ++j ) {
			double x = s_->getStoichEntry(i, j);
			gsl_matrix_set (N, i, j, x);
			gsl_matrix_set (LU_, i, j, x );
		}
	}

	rank_ = myGaussianDecomp( LU_ );

	unsigned int nConsv = nVarMols_ - rank_;
	if ( nConsv == 0 ) {
		cout << "SteadyState::setupSSmatrix(): Number of conserved species == 0. Aborting\n";
		return;
	}
	
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

	/*
	cout << "S = (";
	for ( unsigned int j = 0; j < nVarMols_; ++j )
		cout << s_->S()[ j ] << ", ";
	cout << "), Sinit = ( ";
	for ( unsigned int j = 0; j < nVarMols_; ++j )
		cout << s_->Sinit()[ j ] << ", ";
	cout << ")\n";
	*/

	for ( unsigned int i = 0; i < nConsv; ++i )
		for ( unsigned int j = 0; j < nVarMols_; ++j )
			total_[i] += gsl_matrix_get( gamma_, i, j ) * s_->S()[ j ];

	gsl_matrix_free( N );

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

void SteadyState::classifyState( const double* T )
{
	// unsigned int nConsv = nVarMols_ - rank_;
	gsl_matrix* J = gsl_matrix_calloc ( nVarMols_, nVarMols_ );
	// double* yprime = new double[ nVarMols_ ];
	// vector< double > yprime( nVarMols_, 0.0 );
	// Generate an approximation to the Jacobean by generating small
	// increments to each of the molecules in the steady state, one 
	// at a time, and putting the resultant rate vector into a column
	// of the J matrix.
	// This needs a bit of heuristic to decide what is a 'small' increment.
	// Use the CoInits for this. Stoichiometry shouldn't matter too much.
	// I used the totals from consv rules earlier, but that can have 
	// negative values.
	double tot = 0.0;
	for ( unsigned int i = 0; i < nVarMols_; ++i ) {
		tot += s_->S()[i];
	}
	tot *= DELTA;
	
	// Fill up Jacobian
	for ( unsigned int i = 0; i < nVarMols_; ++i ) {
		double orig = s_->S()[i];
		if ( isnan( orig ) ) {
			cout << "Warning: SteadyState::classifyState: orig=nan\n";
			solutionStatus_ = 2; // Steady state OK, eig failed
			gsl_matrix_free ( J );
			return;
		}
		if ( isnan( tot ) ) {
			cout << "Warning: SteadyState::classifyState: tot=nan\n";
			solutionStatus_ = 2; // Steady state OK, eig failed
			gsl_matrix_free ( J );
			return;
		}
		s_->S()[i] = orig + tot;
		s_->updateV();
		s_->S()[i] = orig;
	// 	yprime.assign( nVarMols_, 0.0 )
	// 	vector< double >::iterator y = yprime.begin();

		// Compute the rates for each mol.
		for ( unsigned int j = 0; j < nVarMols_; ++j ) {
	//		*y++ = N_.computeRowRate( j, s_->velocity() );
			double temp = s_->N().computeRowRate( j, s_->velocity() );
			gsl_matrix_set( J, i, j, temp );
		}
	}

	// Jacobian is now ready. Find eigenvalues.
	gsl_vector_complex* vec = gsl_vector_complex_alloc( nVarMols_ );
	gsl_eigen_nonsymm_workspace* workspace =
		gsl_eigen_nonsymm_alloc( nVarMols_ );
	int status = gsl_eigen_nonsymm( J, vec, workspace );
	eigenvalues_.clear();
	eigenvalues_.resize( nVarMols_, 0.0 );
	if ( status != GSL_SUCCESS ) {
		cout << "Warning: SteadyState::classifyState failed to find eigenvalues. Status = " <<
			status << endl;
		solutionStatus_ = 2; // Steady state OK, eig classification failed
	} else { // Eigenvalues are ready. Classify state.
		nNegEigenvalues_ = 0;
		nPosEigenvalues_ = 0;
		for ( unsigned int i = 0; i < nVarMols_; ++i ) {
			gsl_complex z = gsl_vector_complex_get( vec, i );
			double r = GSL_REAL( z );
			nNegEigenvalues_ += ( r < -EPSILON );
			nPosEigenvalues_ += ( r > EPSILON );
			eigenvalues_[i] = r;
			// We have a problem here because nVarMols_ usually > rank
			// This means we have several zero eigenvalues.
		}

		if ( nNegEigenvalues_ == rank_ ) 
			stateType_ = 0; // Stable
		else if ( nPosEigenvalues_ == rank_ ) // Never see it.
			stateType_ = 1; // Unstable
		else  if (nPosEigenvalues_ == 1)
			stateType_ = 2; // Saddle
		else if ( nPosEigenvalues_ >= 2 )
			stateType_ = 3; // putative oscillatory
		else if ( nNegEigenvalues_ == ( rank_ - 1) && nPosEigenvalues_ == 0 )
			stateType_ = 4; // one zero or unclassified eigenvalue. Messy.
		else
			stateType_ = 5; // Other
	}

	gsl_vector_complex_free( vec );
	gsl_matrix_free ( J );
	gsl_eigen_nonsymm_free( workspace );
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
	double * T = (double *) calloc( nConsv, sizeof( double ) );

	unsigned int i, j;

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

	vector< double > repair( nVarMols_, 0.0 );
	for ( unsigned int j = 0; j < nVarMols_; ++j )
		repair[j] = s_->S()[j];

	int status = iterate( gsl_multiroot_fsolver_hybrids, &ri, maxIter_ );
	if ( status ) // It failed. Fall back with the Newton method
		status = iterate( gsl_multiroot_fsolver_dnewton, &ri, maxIter_ );
	status_ = string( gsl_strerror( status ) );
	nIter_ = ri.nIter;
	if ( status == GSL_SUCCESS ) {
		solutionStatus_ = 0; // Good solution
		classifyState( T );
		/*
		 * Should happen in the ss_func.
		for ( i = 0; i < nVarMols_; ++i )
			s_->S()[i] = gsl_vector_get( op( solver->x ), i );
			*/
	} else {
		cout << "Warning: SteadyState iteration failed, status = " <<
			status_ << ", nIter = " << nIter_ << endl;
		// Repair the mess
		for ( unsigned int j = 0; j < nVarMols_; ++j )
			s_->S()[j] = repair[j];
		solutionStatus_ = 1; // Steady state failed.
	}

	// Clean up.
	free( T );
}

int ss_func( const gsl_vector* x, void* params, gsl_vector* f )
{
	struct reac_info* ri = (struct reac_info *)params;
	int num_consv = ri->num_mols - ri->rank;
	Stoich* s = ri->s;

	for ( int i = 0; i < ri->num_mols; ++i ) {
		double temp = op( gsl_vector_get( x, i ) );
		if ( isnan( temp ) || isinf( temp ) ) { 
			return GSL_ERANGE;
		} else {
			s->S()[i] = temp;
		}
	}
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

/**
 * eliminateRowsBelow:
 * Eliminates leftmost entry of all rows below 'start'.
 * Returns the row index of the row which has the leftmost nonzero
 * entry. If this sticks out beyond numReacs, then the elimination is
 * complete, and it returns a zero to indicate this.
 * In leftCol it returns the column # of this leftmost nonzero entry.
 * Zeroes out anything below EPSILON.
 */
void eliminateRowsBelow( gsl_matrix* U, int start, int leftCol )
{
	int numMols = U->size1;
	double pivot = gsl_matrix_get( U, start, leftCol );
	assert( fabs( pivot ) > SteadyState::EPSILON );
	for ( int i = start + 1; i < numMols; ++i ) {
		double factor = gsl_matrix_get( U, i, leftCol );
		if ( fabs ( factor ) > SteadyState::EPSILON ) {
			factor = factor / pivot;
			for ( size_t j = leftCol + 1; j < U->size2; ++j ) {
				double x = gsl_matrix_get( U, i, j );
				double y = gsl_matrix_get( U, start, j );
				x -= y * factor;
				if ( fabs( x ) < SteadyState::EPSILON )
					x = 0.0;
				gsl_matrix_set( U, i, j, x  );
			}
		}
		gsl_matrix_set( U, i, leftCol, 0.0 ); // Cleaning up.
	}
}

/**
 * reorderRows:
 * Finds the leftmost row beginning from start, ignoring everything to the
 * left of leftCol. Puts this row at 'start', swapping with the original.
 * Assumes that the matrix is set up as [N I]. 
 * Returns the new value of leftCol
 * If there are no appropriate rows, returns numReacs.
 */
int reorderRows( gsl_matrix* U, int start, int leftCol )
{
	int leftMostRow = start;
	int numReacs = U->size2 - U->size1;
	int newLeftCol = numReacs;
	for ( size_t i = start; i < U->size1; ++i ) {
		for ( int j = leftCol; j < numReacs; ++j ) {
			if ( fabs( gsl_matrix_get( U, i, j ) ) > SteadyState::EPSILON ){
				if ( j < newLeftCol ) {
					newLeftCol = j;
					leftMostRow = i;
				}
				break;
			}
		}
	}
	if ( leftMostRow != start ) { // swap them.
		gsl_matrix_swap_rows( U, start, leftMostRow );
	}
	return newLeftCol;
}

/**
 * Does a simple gaussian decomposition. Assumes U has nice clean numbers
 * so I can apply a generous EPSILON to zero things out.
 * Assumes that the U matrix is the N matrix padded out by an identity 
 * matrix on the right.
 * Returns rank.
 */
int myGaussianDecomp( gsl_matrix* U )
{
	int numMols = U->size1;
	int numReacs = U->size2 - numMols;
	int i;
	// Start out with a nonzero entry at 0,0
	int leftCol = reorderRows( U, 0, 0 );

	for ( i = 0; i < numMols - 1; ++i ) {
		eliminateRowsBelow( U, i, leftCol );
		leftCol = reorderRows( U, i + 1, leftCol );
		if ( leftCol == numReacs )
			break;
	}
	return i + 1;
}

//////////////////////////////////////////////////////////////////
// Utility functions for doing scans for steady states
//////////////////////////////////////////////////////////////////

#if 0
/**
 * Checks if this molecule is the last remaining molecule in a
 * conservation block. If so, it has to be assigned the remaing mols
 * returns the consv block index (j). On failure, returns -1.
 */
int SteadyState::isLastConsvMol( int i )
{
	for ( unsigned int j = 0; j < total_.size(); ++j ) {
		// First check if the gamma entry here is nonzero.
		if ( fabs (gsl_matrix_get( gamma_, j, i ) ) < EPSILON ) {
			continue;
		}
		// Go on to check if it is the last nonzero entry.
		bool isLast = 1;
		for ( unsigned int k = i+1; k < nVarMols_; ++k ) {
			if ( fabs (gsl_matrix_get( gamma_, j, k ) ) > EPSILON ) {
				isLast = 0;
				break;
			}
		}
		if ( isLast )
			return j;
	}
	return -1;
}

void SteadyState::recalcRemainingTotal( 
	vector< double >& y, vector< double >& tot )
{
	for ( unsigned int j = 0; j < tot.size() ; ++j ) {
		double temp = 0.0;
		for ( unsigned int i = 0; i < nVarMols_; ++i ) {
			temp += gsl_matrix_get( gamma_, j, i ) * y[i];
		}
		tot[j] = total_[j] - temp;
	}
}

void SteadyState::randomInit()
{
	// double* y = s_->S();
	vector< double > y( nVarMols_, 0.0 );
	vector< double > remainingTotal( total_ );
	double denom = 0.0;
	
	for ( unsigned int i = 0; i < nVarMols_; ++i ) {
		int j = isLastConsvMol( i );
		if ( j >= 0 ) {
			denom = gsl_matrix_get( gamma_, j, i );
			y[i] = remainingTotal[j] / denom;
			assert( y[i] > 0 );
		} else { // Put in another random number here.
			double p = mtrand();
			for ( unsigned int k = 0; k < total_.size(); ++k ) {
				denom = gsl_matrix_get( gamma_, k, i );
				if ( fabs( denom ) > EPSILON ) {
					double x = p * remainingTotal[k] / denom;
					if ( x > EPSILON ) {
						y[ i ] = x;
						break;
					}
				}
			}
		}
		recalcRemainingTotal( y, remainingTotal );
	}
	// Sanity check
	recalcRemainingTotal( y, remainingTotal );
	for ( unsigned int j = 0; j < total_.size(); ++j ) {
		assert( fabs( remainingTotal[j] ) < EPSILON );
	}
	for ( unsigned int i = 0; i < nVarMols_; ++i )
		s_->S()[i] = y[i];
}
#endif

void recalcTotal( vector< double >& tot, gsl_matrix* g, const double* S )
{
	assert( g->size1 == tot.size() );
	for ( unsigned int i = 0; i < g->size1; ++i ) {
		double t = 0.0;
		for ( unsigned int j = 0; j < g->size2; ++j )
			t += gsl_matrix_get( g, i, j ) * S[j];
		tot[ i ] = t;
	}
}

/**
 * Generates a new set of values for the S vector that is a) random
 * and b) obeys the conservation rules.
 */
void SteadyState::randomizeInitialCondition( Eref me )
{
	int numConsv = total_.size();
	recalcTotal( total_, gamma_, s_->S() );
	// The reorderRows function likes to have an I matrix at the end of
	// nVarMols, so we provide space for it, although only its first
	// column is used for the total vector.
	gsl_matrix* U = gsl_matrix_calloc ( numConsv, nVarMols_ + numConsv );
	for ( int i = 0; i < numConsv; ++i ) {
		for ( unsigned int j = 0; j < nVarMols_; ++j ) {
			gsl_matrix_set( U, i, j, gsl_matrix_get( gamma_, i, j ) );
		}
		gsl_matrix_set( U, i, nVarMols_, total_[i] );
	}
	// Do the forward elimination
	int rank = myGaussianDecomp( U );
	assert( rank = numConsv );

	vector< double > eliminatedTotal( numConsv, 0.0 );
	for ( int i = 0; i < numConsv; ++i ) {
		eliminatedTotal[i] = gsl_matrix_get( U, i, nVarMols_ );
	}

	// Put Find a vector Y that fits the consv rules.
	vector< double > y( nVarMols_, 0.0 );
	fitConservationRules( U, eliminatedTotal, y );

	// Sanity check. Try the new vector with the old gamma and tots
	for ( int i = 0; i < numConsv; ++i ) {
		double tot = 0.0;
		for ( unsigned int j = 0; j < nVarMols_; ++j ) {
			tot += y[j] * gsl_matrix_get( gamma_, i, j );
		}
		assert( fabs( tot - total_[i] ) < EPSILON );
	}

	// Put the new values into S.
	// cout << endl;
	for ( unsigned int j = 0; j < nVarMols_; ++j ) {
		s_->S()[j] = y[j];
		// cout << y[j] << " ";
	}
	send0( me, requestYslot ); // Transmit S information to solvers.
	// cout << endl;
}

/**
 * This does the actual work of generating random numbers and
 * making sure they fit.
 */
void SteadyState::fitConservationRules( 
	gsl_matrix* U, const vector< double >& eliminatedTotal,
		vector< double >&y
	)
{
	int numConsv = total_.size();
	int lastJ = nVarMols_;
	for ( int i = numConsv - 1; i >= 0; --i ) {
		for ( unsigned int j = 0; j < nVarMols_; ++j ) {
			double g = gsl_matrix_get( U, i, j );
			if ( fabs( g ) > EPSILON ) {
				// double ytot = calcTot( g, i, j, lastJ );
				double ytot = 0.0;
				for ( int k = j; k < lastJ; ++k ) {
					y[k] = mtrand();
					ytot += y[k] * gsl_matrix_get( U, i, k );
				}
				assert( fabs( ytot ) > EPSILON );
				double lastYtot = 0.0;
				for ( unsigned int k = lastJ; k < nVarMols_; ++k ) {
					lastYtot += y[k] * gsl_matrix_get( U, i, k );
				}
				double scale = ( eliminatedTotal[i] - lastYtot ) / ytot;
				for ( int k = j; k < lastJ; ++k ) {
					y[k] *= scale;
				}
				lastJ = j;
				break;
			}
		}
	}
}

/*
int SteadyState::fitSingleConsvRule( 
	gsl_matrix* U, const vector< double >& eliminatedTotal,
		vector< double >&y, int ruleIndex, int lastJ)
{
	for ( int j = 0; j < nVarMols_; ++j ) {
		g = gsl_matrix_get( U, ruleIndex, j );
		if ( fabs( g ) > EPSILON ) {
			// double ytot = calcTot( g, i, j, lastJ );
			double ytot = 0.0;
			for ( int k = j; k < lastJ; ++k ) {
				y[k] = mtrand();
				ytot += y[k] * gsl_matrix_get( U, ruleIndex, k );
			}
			assert( ytot > 0.0 );
			lastYtot = 0.0;
			for ( int k = lastJ; k < nVarMols_; ++k ) {
				lastYtot += y[k] * gsl_matrix_get( U, ruleIndex, k );
			}
			double scale = ( eliminatedTotal[ruleIndex] - lastYtot )/ytot;
			for ( int k = j; k < lastJ; ++k ) {
				y[k] *= scale;
			}
			return j;
		}
	}
	return 0;
}
*/
