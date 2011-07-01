/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
** Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _MARKOVSOLVER_H
#define _MARKOVSOLVER_H

/////////////////////////////////////////////////////////////
//Class : MarkovSolver
//Author : Vishaka Datta S, 2011, NCBS
//Description : Candidate algorithm for solving the system of equations
//associated with the Markov model of multistate ion channels.   
//
//This implementation computes the matrix exponential using the scaling and
//squaring approach described in 
//"The Scaling and Squaring Method for the Matrix Exponential Revisited", by
//Nicholas J Higham, 2005, Society for Industrial and Applied Mathematics, 
//26(4), pp. 1179-1193.
//
//Any solver for the MarkovChannel class will follow the same implementation.
//MATLAB R2008a uses the same algorithm described above in its expm method. 
//
//After the setup of the MarkovRateTable class, where the user has entered
//the lookup tables for the various transition rates, we have enough
//information to compute all the exponential matrices that correspond to the 
//solution of the kinetic equation at each time step. 
//
//Before the channel simulation is run, the setup of the MarkovSolver requires
//that a table of matrix exponentials be computed and stored. In general, 
//this would require a 2D lookup table where each exponential is index by
//([L],V) where [L] = ligand concentration and V = membrane voltage.  
//In the case all rates are either ligand or voltage dependent, not both, a 1D
//lookup table of exponentials suffices. 
//
//The above computations are achieved by going through the lookup tables 
//of the MarkovRateTable class. In a general case, the number of divisions 
//i.e. step sizes in each lookup table will be different. We choose the smallest
//such step size, and assume that rates with bigger step sizes stay constant
//over this time interval. By iterating over the whole range, we setup the
//exponential table. 
//
//As of now, there is no interpolation used for the terms of the matrix. 
/////////////////////////////////////////////////////////////

///////////////////////////////
//SrcFinfos
///////////////////////////////

class MarkovSolver {
	public : 
	MarkovSolver();

	~MarkovSolver();

	////////////////////////
	//Set-get stuff.
	///////////////////////
	Matrix getQ() const ;
	Vector getState() const;
	Vector getInitialState() const;
	void setInitialState( Vector );

	//Lookup table related stuff. Have stuck to the same naming
	//used in the Interpol2D class for simplicity.
	void setXmin( double );
	double getXmin() const;
	void setXmax( double );
	double getXmax() const;
	void setXdivs( unsigned int );
	unsigned int getXdivs() const;
	double getInvDx() const;

	void setYmin( double );
	double getYmin() const;
	void setYmax( double );
	double getYmax() const;
	void setYdivs( unsigned int );
	unsigned int getYdivs() const;
	double getInvDy() const;
	
	//Computes the updated state of the system. Is called from the process
	//function.
	void computeState();	

	/////////////////////////
	//Lookup table related stuff.
	////////////////////////
	//Fills up lookup table of matrix exponentials.

	void innerFillupTable( MarkovRateTable*, vector< unsigned int >, string, 
											   unsigned int, unsigned int );
	void fillupTable( MarkovRateTable* );

//	Matrix* lookup( double ligandConc = 0, double v );
	/////////////////////////
	//Scaling and squaring algorithm related functions.
	////////////////////////

	//Computes the m'th order Pade Approximant of the Q-matrix. Most of the heavy
	//lifting occurs here.
	Matrix* computePadeApproximant( Matrix*, unsigned int );

	Matrix* computeMatrixExponential();
	
	///////////////////////////
	//MsgDest functions.
	//////////////////////////
	void reinit( const Eref&, ProcPtr );
	void process( const Eref&, ProcPtr );

	//Handles information about Vm from the compartment. 
	void handleVm( double ); 

	//Handles concentration information.
	void handleLigandConc( double );

	//Takes the Id of a MarkovRateTable object to initialize the table of matrix
	//exponentials. 
	void setupTable( Id );

	/////////////
	//Unit test
	////////////
	friend void testMarkovSolver();

	static const Cinfo* initCinfo();

	private :
	//////////////
	//Helper functions.
	/////////////
	
	//Sets the values of xMin, xMax, xDivs, yMin, yMax, yDivs.
	void setLookupParams( MarkovRateTable* );

	//The instantaneous rate matrix.
	Matrix *Q_;

	//////////////
	//Lookup table related stuff.
	/////////////
	/*
	* Exponentials of all rate matrices that are generated over the 
	* duration of the simulation. The idea is that when copies of the channel
	* are made, they will all refer this table to get the appropriate 
	* exponential matrix. 
	*
	* The exponential matrices are computed over a range of voltage levels 
	* and/or ligand concentrations and stored in the appropriate lookup table.
	*
	* Depending on whether
	* 1) All rates are constant,
	* 2) Rates vary with only 1 parameter i.e. ligand/votage,
	* 3) Some rates are 2D i.e. vary with two parameters,
	* we store the table of exponentials in the appropriate pointers below.
	*
	* If a system contains both 2D and 1D rates, then, only the 2D pointer 
	* is used. 
	*/
	//Used for storing exponentials when at least one of the rates are 1D and
	//none are 2D.
	vector< Matrix* > expMats1d_;

	Matrix* expMat_;

	//Used for storing exponentials when at least one of the rates are 2D.
	vector< vector< Matrix* > > expMats2d_;

	double xMin_;
	double xMax_;
	double invDx_;
	unsigned int xDivs_;
	double yMin_;
	double yMax_;
	double invDy_;
	unsigned int yDivs_;

	////////////
	//Miscallenous stuff
	///////////
	
	//Instantaneous state of the system.
	Vector state_;

	//Initial state of the system.
	Vector initialState_;

	//Stands for a lot of things. The dimension of the Q matrix, the number of 
	//states in the rate table, etc which all happen to be the same.
	unsigned int size_;

	//Membrane voltage.
	double Vm_;
	//Ligand concentration.
	double ligandConc_;
};
//End of class definition.

//Matrix exponential related constants. 
//Coefficients of Pade approximants for degrees 3,5,7,9,13.
static double b13[14] = 
			{64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 
			  1187353796428800.0,   129060195264000.0,   10559470521600.0,
						670442572800.0,       33522128640.0,       1323241920.0,
								40840800.0,            960960.0,  16380.0,  182.0,  1.0};

static double b9[10] = 
			{17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 
				30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1 };

static double b7[8] = 
			{17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};

static double b5[6] = {30240, 15120, 3360, 420, 30, 1};

static double b3[4] = {120, 60, 12, 1};

static double thetaM[5] = {1.495585217958292e-2, 2.539398330063230e-1,
	9.504178996162932e-1, 2.097847961257068e0, 5.371920351148152e0};

static unsigned int mCandidates[5] = {3, 5, 7, 9, 13};

#endif
