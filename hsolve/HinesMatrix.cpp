/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**   copyright (C) 2003-2007 Upinder S. Bhalla, Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "HinesMatrix.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include <fstream>
#include <iostream>

#include <vector>
using namespace std;

HinesMatrix::HinesMatrix()
    :
    nCompt_( 0 ),
    dt_( 0.0 ),
    stage_( -1 )
{
    ;
}

void HinesMatrix::setup( const vector< TreeNodeStruct >& tree, double dt )
{
    clear();

    nCompt_ = tree.size();

#if  SANITY_CHECK
    stringstream ss;
    if(nCompt_ <= 0)
    {
        ss << "Horror, horror! Trying to create a matrix with size " << nCompt_
           << endl;
        dump(ss.str(), "ERROR");
        throw range_error("Expected greater than 0.");
    }
#endif     /* -----  not STRICT_CHECK  ----- */

    dt_ = dt;
    tree_ = &tree;

    for ( unsigned int i = 0; i < nCompt_; i++ )
        Ga_.push_back( 2.0 / tree[ i ].Ra );

    makeJunctions();
    makeMatrix();
    makeOperands();

#ifdef USE_CUDA
    allocateMemoryGpu();
    makeCsrMatrixGpu();
#endif
    // Forward flow matrix
    makeForwardFlowMatrix();

    /*
    // Printing swc file of MOOSE numbering.
   printf("Num of compts : %d\n",nCompt_);
   // SWC file
   vector<pair<int,int> > edges;
   // edge information at junctions.
   for(int i=0;i<coupled_.size();i++){
	   int parentId = coupled_[i][coupled_[i].size()-1];
	   for(int j=0;j<coupled_[i].size()-1;j++){
		   edges.push_back(make_pair(coupled_[i][j]+1,parentId+1));
	   }
   	}

   // edge information of branches
   for (int i = 0; i < nCompt_; ++i) {
	   vector<unsigned int> children = tree[i].children;
	   if(children.size() == 1){
		   if(children[0] > i)
			   edges.push_back(make_pair(i+1,children[0]+1));
		   else
			   edges.push_back(make_pair(children[0]+1,i+1));
		}
   	}

	sort(edges.begin(), edges.end());
	edges.insert(edges.begin(), make_pair(nCompt_,-1));

   ofstream swc_file("neuron.swc");
   for(int i=0;i<edges.size();i++){
	   //printf("%d %d\n",edges[i].first, edges[i].second);
	   swc_file << edges[i].first << " " << edges[i].second << endl;
   }
   swc_file.close();


    // Printing stuff
    for ( unsigned int i = 0; i < nCompt_; ++i )
    {
    	vector< unsigned int > c = ( *tree_ )[ i ].children;
    	printf("%d -> ",i);
    	for(int j=0;j<c.size();j++){
    		printf("%d,",c[j]);
    	}
    	printf("\n");
    }

    printf("Coupled data\n");
    for(int i=0;i<coupled_.size();i++){
    	for(int j=0;j<coupled_[i].size();j++){
    		printf("%d,",coupled_[i][j]);
    	}
    	printf("\n");
    }
    printf("Junction Data\n");
    vector< JunctionStruct >::iterator junction = junction_.begin();
    for(;junction != junction_.end();junction++){
    	printf("%d %d\n",junction->index, junction->rank);
    }

    printf("Group Number Data \n");
    map< unsigned int, unsigned int >::iterator groupNo_iter;
    for(groupNo_iter = groupNumber_.begin(); groupNo_iter != groupNumber_.end(); groupNo_iter++){
    	printf("%d,%d \n", groupNo_iter->first, groupNo_iter->second);
    }


    // Printing Ga values
    for (int i = 0; i < nCompt_; ++i) {
		printf("%lf, ", Ga_[i]*100000);
	}
    printf("\n");

    cout << nCompt_ << " " << HJ_.size() << " " << mat_nnz << endl;
    cout << operandBase_.size() << endl;
    */

}
#ifdef USE_CUDA
void HinesMatrix::allocateMemoryGpu(){

}

void HinesMatrix::makeCsrMatrixGpu(){
	// Allocating memory for matrix data
	h_main_diag_passive = new double[nCompt_]();
	h_tridiag_data = new double[3*nCompt_]();

	// Adding passive data to main diagonal
	for(int i=0;i<nCompt_;i++){
		h_main_diag_passive[i] = (*tree_)[i].Cm/(dt_ / 2.0) + 1.0/(*tree_)[i].Rm;
	}

	// From the children data.
	vector<pair<int,double> > non_zero_elements;
	int node1, node2;
	double gi, gj, gij;
	double junction_sum;
	for(int i=0;i<nCompt_;i++){
		// calculating junction sum
		junction_sum = 0;

		// Calculating junction_sum
		vector< unsigned int > branch_nodes;
		branch_nodes.push_back(i);  // pushing the parent to the front of branch_nodes
		branch_nodes.insert(branch_nodes.end(), ( *tree_ )[ i ].children.begin(), ( *tree_ )[ i ].children.end()); // Appending children later

		for(int j=0;j<branch_nodes.size();j++){
			junction_sum += Ga_[branch_nodes[j]];
		}

		// Calculating admittance values and pushing off-diag elements to non-zero set
		for(int j=0;j<branch_nodes.size();j++){
			node1 = branch_nodes[j];

			// Including passive effect to main diagonal elements
			h_main_diag_passive[node1] += Ga_[node1]*(1.0 - Ga_[node1]/junction_sum);

			for(int k=j+1;k<branch_nodes.size();k++){
				node2 = branch_nodes[k];

				gi = Ga_[node1];
				gj = Ga_[node2];
				gij = (gi*gj)/junction_sum;

				non_zero_elements.push_back(make_pair(node1*nCompt_+node2, -1*gij));
				non_zero_elements.push_back(make_pair(node2*nCompt_+node1, -1*gij)); // Adding symmetric element
			}
		}
	}

	// Pushing main diagonal elements to non-zero set
	for(int i=0;i<nCompt_;i++){
		non_zero_elements.push_back(make_pair(i*nCompt_+i, h_main_diag_passive[i]));
	}

	// Number of non zero elements in the matrix.
	int nnz = non_zero_elements.size();

	// Setting up nnz.
	mat_nnz = nnz;

	// Getting elements in csr format
	sort(non_zero_elements.begin(), non_zero_elements.end());

	// Allocating memory for matrix
	h_mat_values = new double[nnz]();
	h_mat_colIndex = new int[nnz]();
	h_mat_rowPtr = new int[nCompt_+1]();
	h_main_diag_map = new int[nCompt_]();
	h_b = new double[nCompt_]();

	// Filling up matrix
	int r,c;
	double value;
	for(int i=0;i<nnz;i++){
		r = non_zero_elements[i].first/nCompt_;
		c = non_zero_elements[i].first%nCompt_;
		value = non_zero_elements[i].second;

		if(r==c){
			// map the index of main diagonal element.
			h_main_diag_map[r] = i;
		}

		switch(c-r){
			case -1:
				h_tridiag_data[r] = value;
				break;
			case 0:
				h_tridiag_data[nCompt_+r] = value;
				break;
			case 1:
				h_tridiag_data[2*nCompt_+r] = value;
				break;
		}

		h_mat_rowPtr[r]++;
		h_mat_colIndex[i] = c;
		h_mat_values[i] = value;
	}

	// Making rowCounts to rowPtr
	int temp;
	int sum = 0;
	// Scan operation on rowPtr;
	for (int i = 0; i < nCompt_+1; ++i)
	{
		temp = h_mat_rowPtr[i];
		h_mat_rowPtr[i] = sum;
		sum += temp;
	}

	// Allocating memory on GPU
	cudaMalloc((void**)&d_mat_values, mat_nnz*sizeof(double));
	cudaMalloc((void**)&d_mat_colIndex, mat_nnz*sizeof(int));
	cudaMalloc((void**)&d_mat_rowPtr, (nCompt_+1)*sizeof(int));
	cudaMalloc((void**)&d_main_diag_map, nCompt_*sizeof(int));
	cudaMalloc((void**)&d_main_diag_passive, nCompt_*sizeof(double));
	cudaMalloc((void**)&d_tridiag_data, 3*nCompt_*sizeof(double));
	cudaMalloc((void**)&d_b, nCompt_*sizeof(double));

	cudaMemcpy(d_mat_values, h_mat_values, mat_nnz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_colIndex, h_mat_colIndex, mat_nnz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_rowPtr, h_mat_rowPtr, (nCompt_+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_main_diag_map, h_main_diag_map, nCompt_*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_main_diag_passive, h_main_diag_passive, nCompt_*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tridiag_data, h_tridiag_data, 3*nCompt_*sizeof(double), cudaMemcpyHostToDevice);


	// Compare two CSR matrices, one from HS_,HJ_ and other from direct method.
	vector<pair<int,double> > non_zeros;

	for(int i=0;i<nCompt_;i++){
		if(HS_[4*i] != 0)
			non_zeros.push_back(make_pair(i*nCompt_+i, HS_[4*i]));

		if(HS_[4*i+1] != 0){
			non_zeros.push_back(make_pair(i*nCompt_+(i+1), HS_[4*i+1]));
			non_zeros.push_back(make_pair((i+1)*nCompt_+i, HS_[4*i+1]));
		}

	}


	// Read off diagonal elements from HJ_
	int comp_num;
	int size;
	int col_ind, col;
	for(int i=0;i<junction_.size();i++){
		comp_num = junction_[i].index;
		size = junction_[i].rank;

		vdIterator elem = operandBase_[comp_num];

		for(int j=0;j<size;j++){
			col_ind = coupled_[groupNumber_[comp_num]].size() - size + j;
			col = coupled_[groupNumber_[comp_num]][col_ind];

			non_zeros.push_back(make_pair(comp_num*nCompt_+col, *elem));
			elem++;
			non_zeros.push_back(make_pair(col*nCompt_+comp_num, *elem));
			elem++;
		}

	}

	cout << non_zeros.size() << " recovered " << mat_nnz << " exists" << endl;

	sort(non_zeros.begin(), non_zeros.end());

	// Check error
	double error = 0;
	double cur_error = 0;
	for(int i=0;i<nCompt_;i++){
		for(int j=h_mat_rowPtr[i];j<h_mat_rowPtr[i+1];j++){
			cur_error = (non_zeros[j].second - h_mat_values[j]);
			error += cur_error;

			//if(cur_error != 0) printf("(%d %d) %lf %lf %lf\n", i, h_mat_colIndex[j], non_zeros[j].second*pow(10,9), h_mat_values[j]*pow(10,9), cur_error*pow(10,9));
		}
	}

	printf("%lf is error\n",error*pow(10,12));


	/*
	// Print passive data
	double main_error = 0;
	double passive_error = 0;
	double right_error = 0;
	int count = 0;
	for(int i=0;i<nCompt_;i++){
		main_error += (HS_[4*i] - h_tridiag_data[nCompt_+i]);
		right_error += (HS_[4*i+1] - h_tridiag_data[2*nCompt_+i]);
		passive_error += (HS_[4*i+2] - h_main_diag_passive[i]);
		//printf("%lf %lf |", HS_[4*i]*100000, h_tridiag_data[nCompt_+i]*100000);
		//printf("%lf %lf |", HS_[4*i+2]*100000, h_main_diag_passive[i]*100000);
		if((HS_[4*i+1] - h_tridiag_data[2*nCompt_+i]) != 0)
			printf("%d %lf %lf |\n", i, HS_[4*i+1]*100000, h_tridiag_data[2*nCompt_+i]*100000);
		if(HS_[4*i+1] != 0)
			count++;


	}
	printf("Errors | %lf | %lf | %lf |\n",right_error*1000000, main_error*1000000, passive_error*1000000);
	printf("count %d \n",count);
	*/


}
#endif

// Printing tri-diagonal system in octave format.
void print_tridiagonal_matrix_system(double* data, int* misplaced_info, int rows){

	double full[rows][rows];

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < rows; ++j)
		{
			full[i][j] = 0;
		}
	}

	for (int i = 0; i < rows; ++i)
	{
		full[i][i] = data[rows+i];
	}

	for (int i = 0; i < rows-1; ++i)
	{
		full[i][misplaced_info[i]] = data[i+1];
		full[misplaced_info[i]][i] = data[i+1];
	}

	cout << "A = [" << endl;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < rows; ++j)
		{
			cout << full[i][j] << ",";
		}
		cout << ";" << endl;
	}
	cout << "]" << endl;

	/*
	cout << "B = [" << endl;
	for (int i = 0; i < rows; ++i)
	{
		cout << rhs[i] << endl;
	}
	cout << "]" << endl;
	*/
}

void HinesMatrix::makeForwardFlowMatrix(){
	ff_system = new double[4*nCompt_]();
	ff_offdiag_mapping = new int[nCompt_]();

	// Setting up passive part of main diagonal
	for(int i=0;i<nCompt_;i++){
		ff_system[2*nCompt_ + i] = (*tree_)[i].Cm/(dt_ / 2.0) + 1.0/(*tree_)[i].Rm;
	}

   // Mapping is nothing but swc file with -1 parent entry.
   // Edge information at junctions.
   for(int i=0;i<coupled_.size();i++){
	   int parentId = coupled_[i][coupled_[i].size()-1];
	   for(int j=0;j<coupled_[i].size()-1;j++){
		   ff_offdiag_mapping[coupled_[i][j]] = parentId;
	   }
	}

   // Edge information of branches
   for (int i = 0; i < nCompt_; ++i) {
	   vector<unsigned int> children = ( *tree_ )[ i ].children;
	   if(children.size() == 1){
		   if(children[0] > i)
			   ff_offdiag_mapping[i] = children[0];
		   else
			   ff_offdiag_mapping[children[0]] = i;
		}
	}

   /*
   // Temporary code
   for(int i=0;i<nCompt_;i++)
	   Ga_[i] = rand()%10+2;
    */

   //// MATRIX construction
	int node1, node2;
	double gi, gj, gij;
	double junction_sum;
   // Contributing junctions to matrix
   for(int i=0;i<coupled_.size();i++){
   	   int parentId = coupled_[i][coupled_[i].size()-1];
   	   junction_sum = 0;
   	   for(int j=0;j<coupled_[i].size();j++){
   		   junction_sum += Ga_[coupled_[i][j]];
   	   }

   	   node1 = parentId;
   	   // Including passive effect to main diagonal elements
   	   ff_system[nCompt_+node1] += Ga_[node1]*(1.0 - Ga_[node1]/junction_sum);

   	   for(int j=0;j<coupled_[i].size()-1;j++){
   		   node2 = coupled_[i][j];

   		   gi = Ga_[node1];
   		   gj = Ga_[node2];
   		   gij = (gi*gj)/junction_sum;

   		   ff_system[nCompt_+node2] += gij;

   		   ff_system[node2+1] = -1*gij;
   	   }
   	}

   // Contributing branches to matrix
   for (int i = 0; i < nCompt_; ++i) {
	   vector<unsigned int> children = ( *tree_ )[ i ].children;
   	   if(children.size() == 1){
   		   if(children[0] > i){
   			   node1 = children[0];
   			   node2 = i;
   		   }else{
   			node1 = i;
   			node2 = children[0];
   		   }

   		   gi = Ga_[node1];
   		   gj = Ga_[node2];
   		   gij = (gi*gj)/(gi+gj);

   		   ff_system[nCompt_+node1] += gij;
   		   ff_system[nCompt_+node2] += gij;

   		   ff_system[node2+1] = -1*gij;
   		}
   	}

   // Verification
   double error = 0;
   double* row_sums = new double[nCompt_]();
   for(int i=0;i<nCompt_;i++){
	   row_sums[i] += ff_system[nCompt_+i];
   }
   for(int i=0;i<nCompt_-1;i++){
	   row_sums[ff_offdiag_mapping[i]] += ff_system[i+1];
	   row_sums[i] += ff_system[i+1];
   }

   for(int i=0;i<nCompt_;i++){
	   //cout << row_sums[i] << endl;
	   error += row_sums[i];
   }
   cout << "Initial matrix error " <<  error << endl;
   //print_tridiagonal_matrix_system(ff_system, ff_offdiag_mapping, nCompt_);
}


void HinesMatrix::clear()
{
    nCompt_ = 0;
    dt_ = 0.0;
    junction_.clear();
    HS_.clear();
    HJ_.clear();
    HJCopy_.clear();
    VMid_.clear();
    operand_.clear();
    backOperand_.clear();
    stage_ = 0;

    tree_ = 0;
    Ga_.clear();
    coupled_.clear();
    operandBase_.clear();
    groupNumber_.clear();
}

bool groupCompare(
    const vector< unsigned int >& A,
    const vector< unsigned int >& B )
{
    if ( A.empty() || B.empty() )
        return 0;

    return A[ 0 ] < B[ 0 ];
}

// Stage 3
void HinesMatrix::makeJunctions()
{
    // 3.1
    for ( unsigned int i = 0; i < nCompt_; ++i )
    {
        const vector< unsigned int >& c = ( *tree_ )[ i ].children;

        if ( c.size() == 0 )
            continue;

        if ( c.size() == 1 )
        {
            int diff = ( int )( c[ 0 ] ) - i;

            if ( diff == 1 || diff == -1 )
                continue;
        }

        // "coupled" contains a list of all children..
        coupled_.push_back( c );
        // ..and the parent compartment itself.
        coupled_.back().push_back( i );
    }

    // 3.2

    vector< vector< unsigned int > >::iterator group;
    for ( group = coupled_.begin(); group != coupled_.end(); ++group )
        sort( group->begin(), group->end() );

    sort( coupled_.begin(), coupled_.end(), groupCompare );


    // 3.3
    unsigned int index;
    unsigned int rank;
    for ( group = coupled_.begin(); group != coupled_.end(); ++group )
        // Loop uptil penultimate compartment in group
        for ( unsigned int c = 0; c < group->size() - 1; ++c )
        {
            index = ( *group )[ c ];
            rank = group->size() - c - 1;
            junction_.push_back( JunctionStruct( index, rank ) );

            groupNumber_[ index ] = group - coupled_.begin();
        }

    sort( junction_.begin(), junction_.end() );
}

// Stage 4
void HinesMatrix::makeMatrix()
{
    const vector< TreeNodeStruct >& node = *tree_;

    // Setting up HS
    HS_.resize( 4 * nCompt_, 0.0 );
    for ( unsigned int i = 0; i < nCompt_; ++i )
        HS_[ 4 * i + 2 ] =
            node[ i ].Cm / ( dt_ / 2.0 ) +
            1.0 / node[ i ].Rm;

    double gi, gj, gij;
    vector< JunctionStruct >::iterator junction = junction_.begin();
    for ( unsigned int i = 0; i < nCompt_ - 1; ++i )
    {
        if ( !junction_.empty() &&
                junction < junction_.end() &&
                i == junction->index )
        {
            ++junction;
            continue;
        }

        gi = Ga_[ i ];
        gj = Ga_[ i + 1 ];
        gij = gi * gj / ( gi + gj );

        HS_[ 4 * i + 1 ] = -gij;
        HS_[ 4 * i + 2 ] += gij;
        HS_[ 4 * i + 6 ] += gij;
    }

    vector< vector< unsigned int > >::iterator group;
    vector< unsigned int >::iterator i;
    for ( group = coupled_.begin(); group != coupled_.end(); ++group )
    {
        double gsum = 0.0;

        for ( i = group->begin(); i != group->end(); ++i )
            gsum += Ga_[ *i ];

        for ( i = group->begin(); i != group->end(); ++i )
        {
            gi = Ga_[ *i ];

            HS_[ 4 * *i + 2 ] += gi * ( 1.0 - gi / gsum );
        }
    }

    // Setting up HJ
    vector< unsigned int >::iterator j;
    unsigned int size = 0;
    unsigned int rank;
    for ( group = coupled_.begin(); group != coupled_.end(); ++group )
    {
        rank = group->size() - 1;
        size += rank * ( rank + 1 );
    }

    HJ_.reserve( size );

    for ( group = coupled_.begin(); group != coupled_.end(); ++group )
    {
        double gsum = 0.0;

        for ( i = group->begin(); i != group->end(); ++i )
            gsum += Ga_[ *i ];

        for ( i = group->begin(); i != group->end() - 1; ++i )
        {
            int base = HJ_.size();

            for ( j = i + 1; j != group->end(); ++j )
            {
                gij = Ga_[ *i ] * Ga_[ *j ] / gsum;

                HJ_.push_back( -gij );
                HJ_.push_back( -gij );
            }

            //~ operandBase_[ *i ] = &HJ_[ base ];
            operandBase_[ *i ] = HJ_.begin() + base;
        }
    }

    // Copy diagonal elements into their final locations
    for ( unsigned int i = 0; i < nCompt_; ++i )
        HS_[ 4 * i ] = HS_[ 4 * i + 2 ];
    // Create copy of HJ
    HJCopy_.assign( HJ_.begin(), HJ_.end() );
}

// Stage 5
void HinesMatrix::makeOperands()
{
    unsigned int index;
    unsigned int rank;
    unsigned int farIndex;
    vdIterator base;
    vector< JunctionStruct >::iterator junction;

    // Allocate space in VMid. Needed, since we will store pointers to its
    // elements below.
    VMid_.resize( nCompt_ );

    // Operands for forward-elimination
    for ( junction = junction_.begin(); junction != junction_.end(); ++junction )
    {
        index = junction->index;
        rank = junction->rank;
        base = operandBase_[ index ];

        // This is the list of compartments connected at a junction.
        const vector< unsigned int >& group =
            coupled_[ groupNumber_[ index ] ];

        if ( rank == 1 )
        {
            operand_.push_back( base );

            // Select last member.
            farIndex = group[ group.size() - 1 ];
            operand_.push_back( HS_.begin() + 4 * farIndex );
            operand_.push_back( VMid_.begin() + farIndex );
        }
        else if ( rank == 2 )
        {
            operand_.push_back( base );

            // Select 2nd last member.
            farIndex = group[ group.size() - 2 ];
            operand_.push_back( HS_.begin() + 4 * farIndex );
            operand_.push_back( VMid_.begin() + farIndex );

            // Select last member.
            farIndex = group[ group.size() - 1 ];
            operand_.push_back( HS_.begin() + 4 * farIndex );
            operand_.push_back( VMid_.begin() + farIndex );
        }
        else
        {
            // Operations on diagonal elements and elements from B (as in Ax = B).
            int start = group.size() - rank;
            for ( unsigned int j = 0; j < rank; ++j )
            {
                farIndex = group[ start + j ];

                // Diagonal elements
                operand_.push_back( HS_.begin() + 4 * farIndex );
                operand_.push_back( base + 2 * j );
                operand_.push_back( base + 2 * j + 1 );

                // Elements from B
                operand_.push_back( HS_.begin() + 4 * farIndex + 3 );
                operand_.push_back( HS_.begin() + 4 * index + 3 );
                operand_.push_back( base + 2 * j + 1 );
            }

            // Operations on off-diagonal elements.
            vdIterator left;
            vdIterator above;
            vdIterator target;

            // Upper triangle elements
            left = base + 1;
            target = base + 2 * rank;
            for ( unsigned int i = 1; i < rank; ++i )
            {
                above = base + 2 * i;
                for ( unsigned int j = 0; j < rank - i; ++j )
                {
                    operand_.push_back( target );
                    operand_.push_back( above );
                    operand_.push_back( left );

                    above += 2;
                    target += 2;
                }
                left += 2;
            }

            // Lower triangle elements
            target = base + 2 * rank + 1;
            above = base;
            for ( unsigned int i = 1; i < rank; ++i )
            {
                left = base + 2 * i + 1;
                for ( unsigned int j = 0; j < rank - i; ++j )
                {
                    operand_.push_back( target );
                    operand_.push_back( above );
                    operand_.push_back( left );

                    /*
                     * This check required because the MS VC++ compiler is
                     * paranoid about iterators going out of bounds, even if
                     * they are never used after that.
                     */
                    if ( i == rank - 1 && j == rank - i - 1 )
                        continue;

                    target += 2;
                    left += 2;
                }
                above += 2;
            }
        }
    }

    // Operands for backward substitution
    for ( junction = junction_.begin(); junction != junction_.end(); ++junction )
    {
        if ( junction->rank < 3 )
            continue;

        index = junction->index;
        rank = junction->rank;
        base = operandBase_[ index ];

        // This is the list of compartments connected at a junction.
        const vector< unsigned int >& group =
            coupled_[ groupNumber_[ index ] ];

        unsigned int start = group.size() - rank;
        for ( unsigned int j = 0; j < rank; ++j )
        {
            farIndex = group[ start + j ];

            backOperand_.push_back( base + 2 * j );
            backOperand_.push_back( VMid_.begin() + farIndex );
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Public interface to matrix
///////////////////////////////////////////////////////////////////////////
unsigned int HinesMatrix::getSize() const
{
    return nCompt_;
}

double HinesMatrix::getA( unsigned int row, unsigned int col ) const
{
    /*
     * If forward elimination is done, or backward substitution is done, and
     * if (row, col) is in the lower triangle, then return 0.
     */
    if ( ( stage_ == 1 || stage_ == 2 ) && row > col )
        return 0.0;

    if ( row >= nCompt_ || col >= nCompt_ )
        return 0.0;

    if ( row == col )
        return HS_[ 4 * row ];

    unsigned int smaller = row < col ? row : col;
    unsigned int bigger = row > col ? row : col;

    if ( groupNumber_.find( smaller ) == groupNumber_.end() )
    {
        if ( bigger - smaller == 1 )
            return HS_[ 4 * smaller + 1 ];
        else
            return 0.0;
    }
    else
    {
        // We could use: groupNumber = groupNumber_[ smaller ], but this is a
        // const function
        unsigned int groupNumber = groupNumber_.find( smaller )->second;
        const vector< unsigned int >& group = coupled_[ groupNumber ];
        unsigned int location, size;
        unsigned int smallRank, bigRank;

        if ( find( group.begin(), group.end(), bigger ) != group.end() )
        {
            location = 0;
            for ( int i = 0; i < static_cast< int >( groupNumber ); ++i )
            {
                size = coupled_[ i ].size();
                location += size * ( size - 1 );
            }

            size = group.size();
            smallRank = group.end() - find( group.begin(), group.end(), smaller ) - 1;
            bigRank = group.end() - find( group.begin(), group.end(), bigger ) - 1;
            location += size * ( size - 1 ) - smallRank * ( smallRank + 1 );
            location += 2 * ( smallRank - bigRank - 1 );

            if ( row == smaller )
                return HJ_[ location ];
            else
                return HJ_[ location + 1 ];
        }
        else
        {
            return 0.0;
        }
    }
}

double HinesMatrix::getB( unsigned int row ) const
{
    return HS_[ 4 * row + 3 ];
}

double HinesMatrix::getVMid( unsigned int row ) const
{
    return VMid_[ row ];
}

///////////////////////////////////////////////////////////////////////////
// Inserting into a stream
///////////////////////////////////////////////////////////////////////////
ostream& operator <<( ostream& s, const HinesMatrix& m )
{
    unsigned int size = m.getSize();

    s << "\nA:\n";
    for ( unsigned int i = 0; i < size; i++ )
    {
        for ( unsigned int j = 0; j < size; j++ )
            s << setw( 12 ) << setprecision( 5 ) << m.getA( i, j );
        s << "\n";
    }

    s << "\n" << "V:\n";
    for ( unsigned int i = 0; i < size; i++ )
        s << m.getVMid( i ) << "\n";

    s << "\n" << "B:\n";
    for ( unsigned int i = 0; i < size; i++ )
        s << m.getB( i ) << "\n";

    return s;
}

///////////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS

#include "TestHSolve.h"

void testHinesMatrix()
{
    vector< int* > childArray;
    vector< unsigned int > childArraySize;

    /**
     *  We test if the Hines' matrix is correctly setup for the following cell:
     *
     *   Soma--->  15 - 14 - 13 - 12
     *              |    |
     *              |    L 11 - 10
     *              |
     *              L 16 - 17 - 18 - 19
     *                      |
     *                      L 9 - 8 - 7 - 6 - 5
     *                      |         |
     *                      |         L 4 - 3
     *                      |
     *                      L 2 - 1 - 0
     *
     *  The numbers are the hines indices of compartments. Compartment X is the
     *  child of compartment Y if X is one level further away from the soma (#15)
     *  than Y. So #17 is the parent of #'s 2, 9 and 18.
     */

    int childArray_1[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1, 0,
        /* c2  */  -1, 1,
        /* c3  */  -1,
        /* c4  */  -1, 3,
        /* c5  */  -1,
        /* c6  */  -1, 5,
        /* c7  */  -1, 4, 6,
        /* c8  */  -1, 7,
        /* c9  */  -1, 8,
        /* c10 */  -1,
        /* c11 */  -1, 10,
        /* c12 */  -1,
        /* c13 */  -1, 12,
        /* c14 */  -1, 11, 13,
        /* c15 */  -1, 14, 16,
        /* c16 */  -1, 17,
        /* c17 */  -1, 2, 9, 18,
        /* c18 */  -1, 19,
        /* c19 */  -1,
    };

    childArray.push_back( childArray_1 );
    childArraySize.push_back( sizeof( childArray_1 ) / sizeof( int ) );

    /**
     *  Cell 2:
     *
     *             3
     *             |
     *   Soma--->  2
     *            / \
     *           /   \
     *          1     0
     *
     */

    int childArray_2[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1,
        /* c2  */  -1, 0, 1, 3,
        /* c3  */  -1,
    };

    childArray.push_back( childArray_2 );
    childArraySize.push_back( sizeof( childArray_2 ) / sizeof( int ) );

    /**
     *  Cell 3:
     *
     *             3
     *             |
     *             2
     *            / \
     *           /   \
     *          1     0  <--- Soma
     *
     */

    int childArray_3[ ] =
    {
        /* c0  */  -1, 2,
        /* c1  */  -1,
        /* c2  */  -1, 1, 3,
        /* c3  */  -1,
    };

    childArray.push_back( childArray_3 );
    childArraySize.push_back( sizeof( childArray_3 ) / sizeof( int ) );

    /**
     *  Cell 4:
     *
     *             3  <--- Soma
     *             |
     *             2
     *            / \
     *           /   \
     *          1     0
     *
     */

    int childArray_4[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1,
        /* c2  */  -1, 0, 1,
        /* c3  */  -1, 2,
    };

    childArray.push_back( childArray_4 );
    childArraySize.push_back( sizeof( childArray_4 ) / sizeof( int ) );

    /**
     *  Cell 5:
     *
     *             1  <--- Soma
     *             |
     *             2
     *            / \
     *           4   0
     *          / \
     *         3   5
     *
     */

    int childArray_5[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1, 2,
        /* c2  */  -1, 0, 4,
        /* c3  */  -1,
        /* c4  */  -1, 3, 5,
        /* c5  */  -1,
    };

    childArray.push_back( childArray_5 );
    childArraySize.push_back( sizeof( childArray_5 ) / sizeof( int ) );

    /**
     *  Cell 6:
     *
     *             3  <--- Soma
     *             L 4
     *               L 6
     *               L 5
     *               L 2
     *               L 1
     *               L 0
     *
     */

    int childArray_6[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1,
        /* c2  */  -1,
        /* c3  */  -1, 4,
        /* c4  */  -1, 0, 1, 2, 5, 6,
        /* c5  */  -1,
        /* c6  */  -1,
    };

    childArray.push_back( childArray_6 );
    childArraySize.push_back( sizeof( childArray_6 ) / sizeof( int ) );

    /**
     *  Cell 7: Single compartment
     */

    int childArray_7[ ] =
    {
        /* c0  */  -1,
    };

    childArray.push_back( childArray_7 );
    childArraySize.push_back( sizeof( childArray_7 ) / sizeof( int ) );

    /**
     *  Cell 8: 3 compartments; soma is in the middle.
     */

    int childArray_8[ ] =
    {
        /* c0  */  -1,
        /* c1  */  -1, 0, 2,
        /* c2  */  -1,
    };

    childArray.push_back( childArray_8 );
    childArraySize.push_back( sizeof( childArray_8 ) / sizeof( int ) );

    /**
     *  Cell 9: 3 compartments; first compartment is soma.
     */

    int childArray_9[ ] =
    {
        /* c0  */  -1, 1,
        /* c1  */  -1, 2,
        /* c2  */  -1,
    };

    childArray.push_back( childArray_9 );
    childArraySize.push_back( sizeof( childArray_9 ) / sizeof( int ) );

    ////////////////////////////////////////////////////////////////////////////
    // Run tests
    ////////////////////////////////////////////////////////////////////////////
    HinesMatrix H;
    vector< TreeNodeStruct > tree;
    double dt = 1.0;

    /*
     * This is the full reference matrix which will be compared to its sparse
     * implementation.
     */
    vector< vector< double > > matrix;

    double epsilon = 1e-17;
    unsigned int i;
    unsigned int j;
    unsigned int nCompt;
    int* array;
    unsigned int arraySize;
    for ( unsigned int cell = 0; cell < childArray.size(); cell++ )
    {
        array = childArray[ cell ];
        arraySize = childArraySize[ cell ];
        nCompt = count( array, array + arraySize, -1 );

        // Prepare cell
        tree.clear();
        tree.resize( nCompt );
        for ( i = 0; i < nCompt; ++i )
        {
            tree[ i ].Ra = 15.0 + 3.0 * i;
            tree[ i ].Rm = 45.0 + 15.0 * i;
            tree[ i ].Cm = 500.0 + 200.0 * i * i;
        }

        int count = -1;
        for ( unsigned int a = 0; a < arraySize; a++ )
            if ( array[ a ] == -1 )
                count++;
            else
                tree[ count ].children.push_back( array[ a ] );

        // Prepare local matrix
        makeFullMatrix( tree, dt, matrix );

        // Prepare sparse matrix
        H.setup( tree, dt );

        // Compare matrices
        for ( i = 0; i < nCompt; ++i )
            for ( j = 0; j < nCompt; ++j )
            {
                ostringstream error;
                error << "Testing Hines' Matrix: Cell# "
                      << cell + 1 << ", entry (" << i << ", " << j << ")";
                ASSERT(
                    fabs( matrix[ i ][ j ] - H.getA( i, j ) ) < epsilon,
                    error.str()
                );
            }
    }

}

#endif // DO_UNIT_TESTS
