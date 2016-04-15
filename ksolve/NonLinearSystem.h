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
#include <limits>

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

    NonlinearSystem( size_t systemSize ) : size( systemSize )
    {
        value.resize( size, 0);
        argument.resize( size, 0 );

        jacobian.resize( size, size, 0);
        invJacobian.resize( size, size, 0);

        x2.resize( size, 0);
        x1.resize( size, 0);

        ri.nVec.resize( size );
    }

    vector_type compute_at(const vector_type& x)
    {
        vector_type result( size );
        system(x, result);
        return result;
    }

    void compute_jacobian( const vector_type& x )
    {

#ifdef  DEBUG
        cout  << "Debug: computing jacobian at " << x << endl;
#endif     /* -----  not DEBUG  ----- */
        double step = 2 * std::numeric_limits< value_type >::min();
        for( size_t i = 0; i < size; i++)
            for( size_t j = 0; j < size; j++)
            {
                vector_type temp = x;
                temp[j] += step;
                system( temp, x2 ); 
                system( x, x1 );
                value_type dvalue = (x2[i] - x1[i])/ step;
                jacobian(i, j) = dvalue;
            }

        // Keep the inverted jacobian ready
        inverse( jacobian, invJacobian );

#ifdef  DEBUG
        cout  << "Debug: " << to_string( ) << endl;
#endif     /* -----  not DEBUG  ----- */
    }

    template<typename T>
    void initialize( const T& x )
    {
        vector_type init;
        init.resize(size, 0);

        for( size_t i = 0; i < size; i++)
            init[i] = x[i];

        argument = init;
        value = compute_at( init );
        compute_jacobian( init );
    }

    string to_string( )
    {
        stringstream ss;

        ss << "=======================================================";
        ss << endl << setw(25) << "State of system: " ;
        ss << " Argument: " << argument << " Value : " << value;
        ss << endl << setw(25) << "Jacobian: " << jacobian;
        ss << endl << setw(25) << "Inverse Jacobian: " << invJacobian;
        ss << endl;
        return ss.str();
    }

    int system( const vector_type& x, vector_type& f )
    {
        int num_consv = ri.num_mols - ri.rank;

        for ( size_t i = 0; i < ri.num_mols; ++i )
        {
            double temp = x[i] * x[i] ;
            if ( isNaN( temp ) || isInfinity( temp ) )
            {
                return ERANGE;
            }
            else
            {
                ri.nVec[i] = temp;
            }
        }
        vector< double > vels;

        ri.pool->updateReacVelocities( &ri.nVec[0], vels );
        assert( vels.size() == static_cast< unsigned int >( ri.num_reacs ) );

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
                dT += ri.gamma( i, j) * (x[j] * x[j]);

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
     * @return  If successful, return true. Check the variable `argument` at
     * which the system value is close to zero (within  the tolerance).
     */
    bool find_roots_gnewton( 
            double tolerance = 1e-10
            , size_t max_iter = 500
            )
    {
        double norm2OfDiff = 1.0;
        size_t iter = 0;
        cerr << "Debug: Starting with " << argument << endl;
        while( ublas::norm_2(value) > tolerance and iter <= max_iter)
        {
            compute_jacobian( argument );
            iter += 1;
            value = compute_at( argument );

            ublas::vector<value_type> s = argument - ublas::prod( invJacobian, value );

#ifdef DEUBG
            cerr << "Previous " << argument << " Next : " << s << endl;
#endif
            argument = s;
            for( size_t ii = 0; ii < size; ii ++)
                ri.nVec[ii] = argument[ii];
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

    vector_type value;
    vector_type argument;
    matrix_type jacobian;
    matrix_type invJacobian;

    // These vector keeps the temporary state computation.
    vector_type x2, x1;

    const size_t size;
    
    ReacInfo ri;

};
