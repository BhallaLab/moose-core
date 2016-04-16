/*
 * =====================================================================================
 *
 *       Filename:  multi_dimensional_root_finding_using_boost.cpp
 *
 *    Description:  Compute root of a multi-dimensional system using boost
 *    libraries.
 *
 *        Version:  1.0
 *        Created:  04/13/2016 11:31:37 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#include <iostream>
#include <sstream>
#include <functional>
#include <cerrno>
#include <iomanip>
#include <algorithm>

// Boost ublas library of matrix algebra.
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


#include "VoxelPools.h"

using namespace std;
using namespace boost::numeric;

typedef double value_type;
typedef ublas::vector<value_type> vector_type;
typedef ublas::matrix<value_type> matrix_type;
typedef function<value_type( const vector_type&  )> equation_type;

class ReacInfo
{
public:
    int rank;
    int num_reacs;
    size_t num_mols;
    int nIter;
    double convergenceCriterion;
    double* T;
    VoxelPools* pool;
    vector< double > nVec;
    ublas::matrix< value_type > Nr;
    ublas::matrix< value_type > gamma;
};


/* Matrix inversion routine.
   Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
bool inverse(const ublas::matrix<T>& input, ublas::matrix<T>& inverse) 
{
    using namespace boost::numeric::ublas;
    typedef permutation_matrix<std::size_t> pmatrix;
    // create a working copy of the input
    matrix<T> A(input);
    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = lu_factorize(A,pm);
    if( res != 0 ) return false;

    // create identity matrix of "inverse"
    inverse.assign(ublas::identity_matrix<T>(A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(A, pm, inverse);

    return true;
}

// A sysmte of non-linear equations. Store the values in result.
class NonlinearSystem
{
public:

    NonlinearSystem( size_t systemSize ) : size_( systemSize )
    {
        f_.resize( size_, 0);
        slopes_.resize( size_, 0);
        x_.resize( size_, 0 );

        J_.resize( size_, size_, 0);
        invJ_.resize( size_, size_, 0);

        x2.resize( size_, 0);
        x1.resize( size_, 0);

        ri.nVec.resize( size_ );
    }

    vector_type compute_at(const vector_type& x)
    {
        vector_type result( size_ );
        system(x, result);
        return result;
    }

    void apply( )
    {
        system(x_, f_);
    }

    int compute_jacobians( const vector_type& x, bool compute_inverse = true )
    {
        for( size_t i = 0; i < size_; i++)
            for( size_t j = 0; j < size_; j++)
            {
                vector_type temp = x;
                temp[j] += step_;
                system( temp, x2 ); 
                system( x, x1 );
                value_type df_ = (x2[i] - x1[i]) / step_;

                // if( std::isnan( df_ ) || std::isinf( df_ ) )
                // {
                //     is_jacobian_valid_ = false;
                //     return ERANGE;
                // }

                J_(i, j) = df_;
            }

        // is_jacobian_valid_ = true;
        // Keep the inverted J_ ready
        //if(is_jacobian_valid_ and compute_inverse )
        if( compute_inverse )
            inverse( J_, invJ_ );

        //cout  << "Debug: " << to_string( ) << endl;
        return 0;
    }

    template<typename T>
    void initialize( const T& x )
    {
        vector_type init;
        init.resize(size_, 0);

        for( size_t i = 0; i < size_; i++)
            init[i] = x[i];

        x_ = init;
        apply();

        compute_jacobians( init );
    }

    string to_string( )
    {
        stringstream ss;

        ss << "=======================================================";
        ss << endl << setw(25) << "State of system: " ;
        ss << " Argument: " << x_ << " Value : " << f_;
        ss << endl << setw(25) << "Jacobian: " << J_;
        ss << endl << setw(25) << "Inverse Jacobian: " << invJ_;
        ss << endl;
        return ss.str();
    }

    int system( const vector_type& x, vector_type& f )
    {
        int num_consv = ri.num_mols - ri.rank;

        for ( size_t i = 0; i < ri.num_mols; ++i )
        {
            double temp = x[i] * x[i] ;

            // if overflow
            if ( std::isnan( temp ) || std::isinf( temp ) )
            {
                cerr << "here with temp " << temp << endl;
                return ERANGE;
            }

            ri.nVec[i] = temp;
        }
        vector< double > vels;

        ri.pool->updateReacVelocities( &ri.nVec[0], vels );
        assert( vels.size_() == static_cast< unsigned int >( ri.num_reacs ) );

        // y = Nr . v
        // Note that Nr is row-echelon: diagonal and above.
        for ( int i = 0; i < ri.rank; ++i )
        {
            double temp = 0;
            for ( int j = i; j < ri.num_reacs; ++j )
                temp += ri.Nr(i, j ) * vels[j];
            f[i] = temp ;
        }

        // dT = gamma.S - T
        for ( int i = 0; i < num_consv; ++i )
        {
            double dT = - ri.T[i];

            for ( size_t  j = 0; j < ri.num_mols; ++j )
            {
                // if overflow
                double temp = x[j] * x[j];
                if ( std::isnan( temp ) || std::isinf( temp ) )
                    return ERANGE;

                dT += ri.gamma( i, j) * temp;
            }
            f[ i + ri.rank] = dT ;
        }

        return 0;
    }


    /**
     * @brief Find roots using Newton-Raphson method.
     *
     * @param tolerance  Default to 1e-12
     * @param max_iter  Maximum number of iteration allowed , default 100
     *
     * @return  If successful, return true. Check the variable `x_` at
     * which the system f_ is close to zero (within  the tolerance).
     */
    bool find_roots_gnewton( 
            double tolerance = 1e-16
            , size_t max_iter = 50
            )
    {
        double norm2OfDiff = 1.0;
        size_t iter = 0;
        while( ublas::norm_2(f_) > tolerance and iter <= max_iter)
        {
            iter += 1;
            cerr << "| " << x_ << endl;
            // Compute the jacoboian at this input.
            compute_jacobians( x_ );
            if( ! is_jacobian_valid_ )
            {
                cerr << "Debug: Jacobian not valid " << endl;
                return false;
            }

            // Compute the f_ of system at this x_, store the f_ in
            // second x_.
            system( x_, f_ );

            // Now compute the next step_. Compute stepSize; if it is zero then
            // we are stuck. Else add it to the current step_.
            vector_type stepSize =  - ublas::prod( invJ_, f_ );
            cerr << "Step  " << stepSize << endl;
            {
                cerr << "Debug: stuck state " << endl;
                cerr << to_string();
                exit(1);
                return false;
            }

            // Update the input to the system by adding the step_ size_.
            x_ +=  stepSize;

            for( size_t ii = 0; ii < size_; ii ++)
                ri.nVec[ii] = x_[ii];
        }

        ri.nIter = iter;

        if( iter > max_iter )
        {
            cerr << "[WARN] Could not find roots of system." << endl;
            cerr <<  "\tTried " << iter << " times." << endl;
            cerr << "\tIteration limits reached" << endl;
            return false;
        }

        cerr << "Info: Computed roots succesfully in " << iter 
            << " iterations " << endl;
        return true;

    }

    /**
     * @brief Compute the slope of function in given dimension.
     *
     * @param which_dimen The index of dimension.
     *
     * @return  Slope.
     */
    value_type slope( unsigned int which_dimen )
    {
        vector_type x = x_;
        x[which_dimen] += step_;
        // x1 and x2 holds the f_ of system at x_ and x (which is x +
        // some step)
        system( x_, x1 );
        system( x, x2 );
        return ublas::norm_2( (x2 - x1)/step_ );
    }

    /** 
     * @brief Suggest the direction to step into.  Take the steepest descent
     * direction.
     *
     * @return  Number of dimension (0 to n-1 ).
     */
    int which_direction_to_stepinto( )
    {
        for( size_t i = 0; i < size_; i++)
            slopes_[i] = slope(i);
        auto iter = std::max_element( slopes_.begin(), slopes_.end());
        return std::distance( slopes_.begin(), iter );
    }

    /**
     * @brief Computes the correction term.
     *
     */
    bool correction_step(  )
    {
        // Get the jacobian at current point. Notice that in this method, we
        // don't have to compute inverse of jacobian

        compute_jacobians( x_, false );
        //cerr << "Jacobian now is " << J_ << endl;
        
        vector_type direction = ublas::prod( J_, x_ );

        // Now take the largest step possible such that the value of system at
        // (x_ - step ) is lower than the value of system as x_.
        vector_type nextState( size_ );

        double diffF = 10.0;
        double factor = 0.13;
        while( true )
        {
            nextState = x_ - (factor * direction);
            diffF = ublas::norm_2( compute_at( nextState )) 
                - ublas::norm_2( compute_at(x_) );

            /** 
             * No need to contnue. Usually we should get negative value.
             */
            if( diffF < 0.0 )
            {
                x_ = nextState;
                return true;
            }

            /** 
             * But we don't want to get caught in infinite loop. So when diffF
             * goes to zero, just terminate. 
             */
            else if( diffF == 0.0 )
            {
#if 0
                cerr << "Warn: Failed to get a good diff. " 
                    << " Diff : " << diffF 
                    << " and factor : " << factor << endl;
                cerr << J_ << endl;
#endif
                return false;
            }

#if 0
            cerr << "Prev: " << x_ << " " 
                << ublas::norm_2( compute_at(x_) ) << endl;
            cerr << "Next: " << nextState << " "
                << ublas::norm_2( compute_at(nextState ) ) << endl;
#endif

            factor = factor / 2.0;
        }
        return true;
    }

    bool find_roots_gradient_descent ( double tolerance = 1e-16 
            , size_t max_iter = 50)
    {
        cerr << "Searching for roots using gradient descent method" << endl;
        
        /*-----------------------------------------------------------------------------
         *  This algorithm has following steps.
         *
         *  while "not satisfied" do
         *      find a good search direction (usually the steepest slope).
         *      step into that direction by "some amount"
         *-----------------------------------------------------------------------------*/
        double startVal = ublas::norm_2( compute_at( x_ ));
        double currentVal;
        cerr << "Starting at " << startVal << endl;
        while( true )
        {
#if 0
            cerr << "Debug: start : " << x_ << " value : " 
                << ublas::norm_2( compute_at( x_ ) ) 
                << endl;
#endif

            if( ! correction_step( ) )
                return false;
            else
                currentVal = ublas::norm_2( compute_at( x_ ));

            // This is a cool solution.
            if( currentVal <= tolerance )
                return true;

            // We are stuck
            if( currentVal == startVal )
                return false;
        }
    }

public:
    const size_t size_;
    double step_ = 1e-6;

    vector_type f_;
    vector_type x_;
    vector_type slopes_;
    matrix_type J_;
    matrix_type invJ_;

    bool is_jacobian_valid_;
    bool is_f_positive_;

    // These vector keeps the temporary state computation.
    vector_type x2, x1;
    
    ReacInfo ri;
};
