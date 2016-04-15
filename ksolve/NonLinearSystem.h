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
        currentPos.resize( size, 0 );

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

    int compute_jacobians( const vector_type& x )
    {
        for( size_t i = 0; i < size; i++)
            for( size_t j = 0; j < size; j++)
            {
                vector_type temp = x;
                temp[j] += step_;
                system( temp, x2 ); 
                system( x, x1 );
                value_type dvalue = (x2[i] - x1[i]) / step_;
                if( std::isnan( dvalue ) || std::isinf( dvalue ) )
                {
                    jacobianValid = false;
                    return ERANGE;
                }
                jacobian(i, j) = dvalue;
            }

        jacobianValid = true;
        // Keep the inverted jacobian ready
        if(jacobianValid)
            inverse( jacobian, invJacobian );

        //cout  << "Debug: " << to_string( ) << endl;
        return 0;
    }

    template<typename T>
    void initialize( const T& x )
    {
        vector_type init;
        init.resize(size, 0);

        for( size_t i = 0; i < size; i++)
            init[i] = x[i];

        currentPos = init;
        compute_jacobians( init );
        if( jacobianValid )
            value = compute_at( init );
    }

    string to_string( )
    {
        stringstream ss;

        ss << "=======================================================";
        ss << endl << setw(25) << "State of system: " ;
        ss << " Argument: " << currentPos << " Value : " << value;
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
     * @return  If successful, return true. Check the variable `currentPos` at
     * which the system value is close to zero (within  the tolerance).
     */
    bool find_roots_gnewton( 
            double tolerance = 1e-16
            , size_t max_iter = 50
            )
    {
        double norm2OfDiff = 1.0;
        size_t iter = 0;
        while( ublas::norm_2(value) > tolerance and iter <= max_iter)
        {
            iter += 1;
            cerr << "| " << currentPos << endl;
            // Compute the jacoboian at this input.
            compute_jacobians( currentPos );
            if( ! jacobianValid )
            {
                cerr << "Debug: Jacobian not valid " << endl;
                return false;
            }

            // Compute the value of system at this currentPos, store the value in
            // second currentPos.
            system( currentPos, value );

            // Now compute the next step_. Compute stepSize; if it is zero then
            // we are stuck. Else add it to the current step_.
            vector_type stepSize =  - ublas::prod( invJacobian, value );
            cerr << "Step  " << stepSize << endl;
            {
                cerr << "Debug: stuck state " << endl;
                cerr << to_string();
                exit(1);
                return false;
            }

            // Update the input to the system by adding the step_ size.
            currentPos +=  stepSize;

            for( size_t ii = 0; ii < size; ii ++)
                ri.nVec[ii] = currentPos[ii];
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
    vector_type currentPos;
    matrix_type jacobian;
    matrix_type invJacobian;

    bool jacobianValid;

    // These vector keeps the temporary state computation.
    vector_type x2, x1;
    double step_ = 1e1;

    const size_t size;
    
    ReacInfo ri;

};
