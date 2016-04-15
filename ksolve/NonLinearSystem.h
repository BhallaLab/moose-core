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

    int compute_jacobians( const vector_type& x )
    {
        for( size_t i = 0; i < size_; i++)
            for( size_t j = 0; j < size_; j++)
            {
                vector_type temp = x;
                temp[j] += step_;
                system( temp, x2 ); 
                system( x, x1 );
                value_type df_ = (x2[i] - x1[i]) / step_;
                if( std::isnan( df_ ) || std::isinf( df_ ) )
                {
                    is_jacobian_valid_ = false;
                    return ERANGE;
                }
                J_(i, j) = df_;
            }

        is_jacobian_valid_ = true;
        // Keep the inverted J_ ready
        if(is_jacobian_valid_)
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
        compute_jacobians( init );
        if( is_jacobian_valid_ )
            f_ = compute_at( init );
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

    value_type slope( size_t which_dimen )
    {
        vector_type x = x_;
        x[which_dimen] += step_;

        // x1 and x2 holds the f_ of system at x_ and x (which is x +
        // some step)
        system( x_, x1 );
        system( x, x2 );
        return ublas::norm_2( x2 - x1 );
    }

    /** 
     * @brief Suggest the direction to step into. 
     *
     * If value of the function is positive at starting point, then we want to
     * descent into the direction of negative slope otherwise we want to go into
     * the direction of positive slope.
     *
     * @return  Number of dimension (0 to n-1 ).
     */
    int which_direction_to_stepinto( )
    {
        for( size_t i = 0; i < size_; i++)
            slopes_[i] = slope(i);

        auto iter = slopes_.begin();

        // FIXME: min and max does not neccessarily mean negative and positive. Let's
        // hope that they are.
        if( is_f_positive_ )
            iter = std::min_element( slopes_.begin(), slopes_.end() );
        else
            iter = std::max( slopes_.begin(), slopes_.end() );

        return std::distance( slopes_.begin(), iter );
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
        for (size_t i = 0; i < size_; i++) 
        {
            cerr << "Slope at " << i << " " << slope( i ) << endl;
        }
        cerr << to_string( ) << endl;
        cerr << which_direction_to_stepinto() << endl;

        exit(1);
    }

public:
    const size_t size_;
    double step_ = 1e1;

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
