/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _HINES_MATRIX_H
#define _HINES_MATRIX_H

#include "StorageFormats.h"

#ifdef DO_UNIT_TESTS
# define ASSERT( isOK, message ) \
	if ( !(isOK) ) { \
		cerr << "\nERROR: Assert '" << #isOK << "' failed on line " << __LINE__ << "\nin file " << __FILE__ << ": " << message << endl; \
		exit( 1 ); \
	} else { \
		cout << ""; \
	}
#else
# define ASSERT( unused, message ) do {} while ( false )
#endif

#include "CudaGlobal.h"

struct JunctionStruct
{
    JunctionStruct( unsigned int i, unsigned int r ) :
        index( i ),
        rank( r )
    {
        ;
    }

    bool operator< ( const JunctionStruct& other ) const
    {
        return ( index < other.index );
    }

    unsigned int index;		///< Hines index of the compartment.
    unsigned int rank;		///< Number of elements "remaining" in this
    ///< compartment's group: i.e., number of children
    ///< with a larger Hines index, +1 for the parent.
};

struct TreeNodeStruct
{
    vector< unsigned int > children;	///< Hines indices of child compts
    double Ra;
    double Rm;
    double Cm;
    double Em;
    double initVm;
};

class HinesMatrix
{
public:
    HinesMatrix();

    void setup( const vector< TreeNodeStruct >& tree, double dt );

    unsigned int getSize() const;
    double getA( unsigned int row, unsigned int col ) const;
    double getB( unsigned int row ) const;
    double getVMid( unsigned int row ) const;

protected:
    typedef vector< double >::iterator vdIterator;

    unsigned int              nCompt_;
    double                    dt_;

    vector< JunctionStruct >  junction_;
    vector< double >          HS_;			/**< Hines, series.
		* Flattened array containing the tridiagonal of the approximately
		* tridiagonal Hines matrix, stacked against the column vector "b" that
		* appears on the RHS of the equation that we're trying to solve: Ax=b.
		*/
    vector< double >          HJ_;			/**< Hines, junctions.
		* Flattened array containing the off-diagonal elements of the Hines
		* matrix */
    vector< double >          HJCopy_;
    vector< double >          VMid_;		///< Compartment voltage at the
    ///< middle of a time step.
    vector< vdIterator >      operand_;
    vector< vdIterator >      backOperand_;
    int                       stage_;		///< Which stage the simulation has
    ///< reached. Used in getA.

#ifdef USE_CUDA
    double* h_mat_values;
    int* h_mat_colIndex;
    int* h_mat_rowPtr;
    int* h_main_diag_map;

    double* h_main_diag_passive;
    double* h_tridiag_data;
    double* h_b;

    double* d_mat_values;
    int* d_mat_colIndex;
    int* d_mat_rowPtr;
    int* d_main_diag_map;

    double* d_main_diag_passive;
    double* d_tridiag_data;
    double* d_b;

    int mat_nnz;
#endif

    //// Forward flow matrix data structures
	/*
	 * ff_system stores the tri-diagonal system as an array of size (4*num_comp)
	 * Column1 - lower diagonal elements
	 * Column2 - main diagonal elements
	 * Column3 - Passive main diagonal elements.
	 * Column4 - RHS
	 */
	double* ff_system;
	/*
	 * Stores the row values of lower off-diagonal elements ordered column wise.
	 */
	int* ff_offdiag_mapping;

	//// Pervasive flow matrix data structures
	coosr_matrix full_mat, upper_mat, lower_mat;
	double* per_rhs;
	double* per_mainDiag_passive;
	int* per_mainDiag_map;

	vector<int> ut_lt_upper, ut_lt_lower, ut_ut_upper, ut_ut_lower;
	int* ut_lt_rowPtr, *ut_ut_rowPtr;
	double* upper_mat_values_copy;
	double* lower_mat_values_copy;

	void print_tridiagonal_matrix_system(double* data, int* misplaced_info, int rows);
	void print_csr_matrix(coosr_matrix &matrix);

private:
    void clear();
    void makeJunctions();
    /**< This function creates junction structs to be stored in junction_.
     *   It does so by first looking through all compartments and finding
     *   those which have more than zero or one child. The zero-one child
     *   compts are left alone. coupled_ is populated with the rest of the
     *   compartments. Each element of coupled_ is sorted (they are all
     *   unsigned ints) and coupled_ itself is sorted by the first element
     *   in each element (groupCompare does this comparison in
     *   HinesMatrix.cpp).
     *   Note: the children themselves are unsigned ints that store the
     *   Hines index of the corresponding child compartment.
     *   So essentially, at each branch, a JunctionStruct is created for
     *   each child, which contains the hines index of that child and its
     *   rank, which is group-size() - childIndex - 1. */
    void makeMatrix();		/**< Populates HS_ and HJ_.
		 *   All of the electrical circuit analysis goes into this one single
		 *   function (and updateMatrix, of course). */
    void makeOperands();	///< Makes operands in order to make forward
    ///< elimination easier.

    const vector< TreeNodeStruct >     *tree_;		///< Stores compt info for
    ///< setup.
    vector< double >                   Ga_;
    vector< vector< unsigned int > >   coupled_;
    /**< Contains a list of all children of a given compt. Also contains
     *   the parent itself. i.e., for each compartment that has more than
     *   one child, coupled_ stores a vector containing the children of the
     *   compartment and the compartment itself.
     *   coupled_ is therefore a vector of such vectors. */
    map< unsigned int, vdIterator >    operandBase_;
    /**< Contains iterators into HJ_ demarcating where a child's neighbours
     *   begin. Used for iterating through HJ_ along with junction_. */
    map< unsigned int, unsigned int >  groupNumber_;
    /**< Tells you the index of a compartment's group within coupled_,
     *   given the compartment's Hines index. */

#ifdef USE_CUDA
    /*
     * Allocates memory for GPU.
     */
    void allocateMemoryGpu();
    /*
     * Creates hines matrix and stores it in CSR format.
     */
    void makeCsrMatrixGpu();
#endif
    /*
	 * Create forward flow hines matrix
	 */
    void makeForwardFlowMatrix();
    /*
     * Create pervasive flow hines matrix
     */
    void makePervasiveFlowMatrix();
    void storePervasiveMatrix(vector<vector<int> > &child_list);

    // helpers for pervasive matrix
    void exclusive_scan(int* data, int rows);
    void generate_coosr_matrix(int num_comp, const vector<pair<long long int,double> > &full_tri, coosr_matrix &full_mat);
    void construct_elimination_information(coosr_matrix full_mat, coosr_matrix upper_mat, coosr_matrix lower_mat,
    			vector<int> &ut_lt_upper, vector<int> &ut_lt_lower, vector<int> &ut_ut_upper, vector<int> &ut_ut_lower ,
    			int* ut_lt_rowPtr, int* ut_ut_rowPtr);

};

#endif // _HINES_MATRIX_H
