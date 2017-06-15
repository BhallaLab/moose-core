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
    /*
     * Data structures for storing matrix in CSR format
     * Refer( https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29) .
     */
    double* h_mat_values;
    int* h_mat_colIndex;
    int* h_mat_rowPtr;
    int* h_main_diag_map; // Stores the index of diagonal element in csr element array.

    double* h_main_diag_passive; // Passive component of main diagonal
    double* h_b; // RHS matrix

    // Corresponding device pointer
    double* d_mat_values;
    int* d_mat_colIndex;
    int* d_mat_rowPtr;
    int* d_main_diag_map;

    double* d_main_diag_passive;
    double* d_b;

    int mat_nnz; // Number of non-zeros in the matrix.
#endif

    /*
     * Forward Flow matrix :
     * When circuit of a compartment has only one Ra component instead of two Ra/2 on either side, the corresponding matrix
     * of multi-compartment model is a forward flow matrix.
     * *** There is only one non-zero element after main diagonal element.
     * *** Symmetric
     */

    //// Forward flow matrix data structure description
	/*
	 * ff_system stores the tri-diagonal system as an array of size (4*num_comp)
	 * Column1 - lower diagonal elements
	 * Column2 - main diagonal elements
	 * Column3 - Passive main diagonal elements.
	 * Column4 - RHS
	 */
	double* ff_system;
	int* ff_offdiag_mapping;// Stores the row values of lower off-diagonal elements ordered column wise.


	/*
	 * Pervasive Flow matrix :
	 * When circuit of a compartment has two Ra/2 on either side instead of one Ra, then the corresponding matrix
	 * of multi-compartment model is a forward flow matrix.
	 * *** Symmetric
	 */
	//// Pervasive flow matrix data structures.
	coosr_matrix qfull_mat;
	double* per_rhs, *per_mainDiag_passive; // RHS of matrix and passive part of main diagonal.
	vector<int> eliminfo_r1, eliminfo_r2; // For a given elimination e(r,c), we store overlapping elements in row c and row r
	// For a given elimination e(r,c), overlapping element of A[r][r] is A[c][r] and (A[c][r] != 0).
	// This array stores the index position of A[c][r] in qfull_mat elements.
	int* eliminfo_diag;
	int* elim_rowPtr; // Pointer to elimination information for each elimination.
	int* upper_triang_offsets; // Positions of first non-zero upper triangular element in qfull_mat elements.
	double* perv_mat_values_copy; // Used for storing a copy of values array in qfull_mat.
	double* perv_dynamic; // Array of size 2*nCompt. First part is main diagonal, second part is RHS.

	// Helper functions
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
     * Creates hines matrix and stores it in CSR format in both CPU and GPU.
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
    void construct_elimination_information_opt(coosr_matrix qfull_mat, vector<int> &eliminfo_r1, vector<int> &eliminfo_r2,
    		int* eliminfo_diag,	int* elim_rowPtr, int* upper_triang_offsets, int num_elims);
};

#endif // _HINES_MATRIX_H
