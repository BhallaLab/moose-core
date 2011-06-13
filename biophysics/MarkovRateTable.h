#ifndef _MARKOVRATETABLE_H
#define _MARKOVRATETABLE_H

//Presents a unified interface to deal with transition rates that are dependent
//on one or two parameters, or are constant.
//For two parameter lookups, this class borrows
//heavily from the lookup functions implemented in the Interpol2D class. The one
//parameter lookup functions are based on the lookup tables used in the HHGate
//class.
//
//The class consists of a table of pointers to the VectorTable class and the
//Interpol2D class. Since these classes are user-defined and possess
//constructors, destructors, etc., using a union is not possible (but would've
//been more efficient).
//
//For a given rate entry (i,j) either one of these pointers may be NULL, not
//both. Also, both pointers for rate (i,j) cannot simultaneously be NON-NULL. 
//
//In case a rate is constant, it is stored in a VectorTable class whose table is
//of size 1. 
//
//In case of no transition, both pointers at (i,j) are null.

template <class T> 
vector< vector< T > > resize( vector< vector< T > >table, unsigned int n, T init )
{
	table.resize( n );	

	for ( unsigned int i = 0; i < n; ++i ) 
		table[i].resize( n, init );		

	return table;
}

class MarkovRateTable {
	public : 	
	MarkovRateTable(); 

	MarkovRateTable( unsigned int );

	//One parameter rate table set and get functions.
	vector< double > getVtChildTable( unsigned int, unsigned int ) const; 
	void setVtChildTable( vector< unsigned int >, vector< double >, vector< double >, bool );

	//Two parameter rate table set and get functions.
	vector< vector< double > > getInt2dChildTable( unsigned int, unsigned int ) const;
	void setInt2dChildTable( vector< unsigned int >, vector< double >, vector< vector< double > > );
	
	//Lookup functions.
	double lookup1D( unsigned int, unsigned int, double );
	double lookup2D( unsigned int, unsigned int, double, double );

	//////////////////
	//Helper functions
	/////////////////
	
	//Returns true if the (i,j)'th rate is zero i.e. no transitions between states
	//i and j. When this is the case, both the  (i,j)'th pointers in vtTables_
	// and int2dTables_ are not set i.e. both are zero.
	bool isRateZero( unsigned int, unsigned int ) const ;

	//Returns true if the (i,j)'th rate is a constant. This is true when the
	//vtTables_[i][j] pointer points to a 1D lookup table of size 1.
	bool isRateConstant( unsigned int, unsigned int ) const ;

	//Returns true if the (i,j)'th rate is varies with only one parameter. 
	bool isRateOneParam( unsigned int, unsigned int ) const ;  

	//Returns true if the (i,j)'th rate is dependent on ligand concentration and
	//not membrane voltage, and is set with a 1D lookup table. 
	bool isRateLigandDep( unsigned int, unsigned int) const;

	//Returns true if the (i,j)'th rate varies with two parameters.
	bool isRateTwoParam( unsigned int, unsigned int ) const ;

	//Returns true if a table is being accessed at an invalid address.
	bool areIndicesOutOfBounds( unsigned int, unsigned int ) const ;

	private : 
	vector< vector< VectorTable* > > vtTables_;  
	vector< vector< Interpol2D* > > int2dTables_; 
	
	vector< vector< bool > > useLigandConc_; 

	unsigned int size_;
};
#endif
