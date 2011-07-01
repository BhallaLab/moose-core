/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <values.h>		//Needed for DBL_MAX and DBL_MIN
#include "header.h"
#include "MatrixOps.h"

#include "VectorTable.h"
#include "../builtins/Interpol2D.h"
#include "MarkovRateTable.h"

#include "MarkovSolver.h"

SrcFinfo1< Vector >* stateOut()
{
	static SrcFinfo1< Vector > stateOut("stateOut",
		"Sends updated state to the MarkovChannel class."
		);
	return &stateOut;
}

const Cinfo* MarkovSolver::initCinfo()
{
	/////////////////////
	//SharedFinfos
	/////////////////////
	static DestFinfo handleVm("handleVm", 
			"Handles incoming message containing voltage information.",
			new OpFunc1< MarkovRateTable, double >(&MarkovRateTable::handleVm)
			);

	static Finfo* channelShared[] = 
	{
		&handleVm
	};

	static SharedFinfo channel("channel",
			"This message couples the MarkovSolver to the Compartment. The "
			"compartment needs Vm in order to look up the correct matrix "
			"exponential for computing the state.",
			channelShared, sizeof( channelShared ) / sizeof( Finfo* ) 
			);

	//////////////////////
	//DestFinfos
	//////////////////////
	
	static DestFinfo process(	"process",
			"Handles process call",
			new ProcOpFunc< MarkovSolver >( &MarkovSolver::process ) ); 

	static DestFinfo reinit( "reinit", 
			"Handles reinit call",
			new ProcOpFunc< MarkovSolver >( &MarkovSolver::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
			"This is a shared message to receive Process message from the"
			"scheduler. The first entry is a MsgDest for the Process "
			"operation. It has a single argument, ProcInfo, which "
			"holds lots of information about current time, thread, dt and"
			"so on. The second entry is a MsgDest for the Reinit "
			"operation. It also uses ProcInfo.",
		processShared, sizeof( processShared ) / sizeof( Finfo* )
	);

	static DestFinfo handleLigandConc("handleLigandConc",
			"Handles incoming message containing ligand concentration.",
			new OpFunc1< MarkovSolver, double >(&MarkovSolver::handleLigandConc) 
			);

	static DestFinfo setuptable("setuptable",
			"Setups the table of matrix exponentials associated with the"
			" solver object.",
			new OpFunc1< MarkovSolver, Id >(&MarkovSolver::setupTable) 
			);

	//////////////////////
	//*ValueFinfos
	/////////////////////
	
	static ReadOnlyValueFinfo< MarkovSolver, Matrix > Q("Q",
			"Instantaneous rate matrix.",
			&MarkovSolver::getQ
			);

	static ReadOnlyValueFinfo< MarkovSolver, Vector > state("state",
			"Current state of the channel.",
			&MarkovSolver::getState 
			);

	static ValueFinfo< MarkovSolver, Vector > initialstate("initialstate",
			"Initial state of the channel.",
			&MarkovSolver::setInitialState,
			&MarkovSolver::getInitialState 
			);

	static ValueFinfo< MarkovSolver, double > xmin( "xmin",
		"Minimum value for x axis of lookup table",
			&MarkovSolver::setXmin,
			&MarkovSolver::getXmin
		);
	static ValueFinfo< MarkovSolver, double > xmax( "xmax",
		"Maximum value for x axis of lookup table",
			&MarkovSolver::setXmax,
			&MarkovSolver::getXmax
		);
	static ValueFinfo< MarkovSolver, unsigned int > xdivs( "xdivs",
		"# of divisions on x axis of lookup table",
			&MarkovSolver::setXdivs,
			&MarkovSolver::getXdivs
		);
	static ReadOnlyValueFinfo< MarkovSolver, double > invdx( "invdx",
		"Reciprocal of increment on x axis of lookup table",
			&MarkovSolver::getInvDx
		);
	static ValueFinfo< MarkovSolver, double > ymin( "ymin",
		"Minimum value for y axis of lookup table",
			&MarkovSolver::setYmin,
			&MarkovSolver::getYmin
		);
	static ValueFinfo< MarkovSolver, double > ymax( "ymax",
		"Maximum value for y axis of lookup table",
			&MarkovSolver::setYmax,
			&MarkovSolver::getYmax
		);
	static ValueFinfo< MarkovSolver, unsigned int > ydivs( "ydivs",
		"# of divisions on y axis of lookup table",
			&MarkovSolver::setYdivs,
			&MarkovSolver::getYdivs
		);
	static ReadOnlyValueFinfo< MarkovSolver, double > invdy( "invdy",
		"Reciprocal of increment on y axis of lookup table",
			&MarkovSolver::getInvDy
		);

	static Finfo* markovSolverFinfos[] = 	
	{
		&channel,						//SharedFinfo	
		&proc,							//SharedFinfo
		stateOut(), 				//SrcFinfo
		&handleLigandConc,	//DestFinfo
		&setuptable,				//DestFinfo
		&Q,									//ReadOnlyValueFinfo
		&state,							//ReadOnlyValueFinfo
		&initialstate,			//ReadOnlyValueFinfo
		&xmin,							//ValueFinfo
		&xmax,							//ValueFinfo
		&xdivs,							//ValueFinfo
		&invdx,							//ReadOnlyValueFinfo
		&ymin,							//ValueFinfo
		&ymax,							//ValueFinfo
		&ydivs,							//ValueFinfo
		&invdy							//ReadOnlyValueFinfo
	};

	static Cinfo markovSolverCinfo(
			"MarkovSolver",			
			Neutral::initCinfo(),
			markovSolverFinfos,
			sizeof( markovSolverFinfos ) / sizeof( Finfo* ),
			new Dinfo< MarkovSolver > 
			);

	return &markovSolverCinfo;
}

static const Cinfo* markovSolverCinfo = MarkovSolver::initCinfo();

MarkovSolver::MarkovSolver() : Q_(0), expMats1d_(0), expMat_(0), 
	expMats2d_(0), xMin_(DBL_MAX), xMax_(DBL_MIN), xDivs_(0u), 
	yMin_(DBL_MAX), yMax_(DBL_MIN), yDivs_(0u), size_(0u), Vm_(0),
 	ligandConc_(0)
{
	;
}

MarkovSolver::~MarkovSolver()
{
	if ( Q_ )
		delete Q_;

	if ( !expMats1d_.empty() )
	{
		while ( !expMats1d_.empty() )
		{
			delete expMats1d_.back();
			expMats1d_.pop_back();
		}
	}

	if ( !expMats2d_.empty() )
	{
		unsigned int n = expMats2d_.size(); 
		for( unsigned int i = 0; i < n; ++i )
		{
			for ( unsigned int j = 0; j < expMats2d_[i].size(); ++j )
				delete expMats2d_[i][j];
		}
	}

	if ( !expMat_ )
		delete expMat_;
}

////////////////////////////////////
//Set-Get functions
///////////////////////////////////
Matrix MarkovSolver::getQ() const
{
	return *Q_;
}

Vector MarkovSolver::getState() const
{
	return state_;
}

Vector MarkovSolver::getInitialState() const
{
	return initialState_;
}

void MarkovSolver::setInitialState( Vector state )
{
	state_ = state;
}

void MarkovSolver::setXmin( double xMin )
{
	xMin_ = xMin;
}

double MarkovSolver::getXmin() const
{
	return xMin_;
}

void MarkovSolver::setXmax( double xMax ) 
{
	xMax_ = xMax;	
}

double MarkovSolver::getXmax() const
{
	return xMax_;
}

void MarkovSolver::setXdivs( unsigned int xDivs )
{
	xDivs_ = xDivs;
}

unsigned int MarkovSolver::getXdivs( ) const {
	return xDivs_;
}

double MarkovSolver::getInvDx() const {
	return invDx_;
}

void MarkovSolver::setYmin( double yMin )
{
	yMin_ = yMin;
}

double MarkovSolver::getYmin() const
{
	return yMin_;
}

void MarkovSolver::setYmax( double yMax )
{
	yMax_ = yMax;
}

double MarkovSolver::getYmax() const
{
	return yMax_;
}

void MarkovSolver::setYdivs( unsigned int yDivs )
{
	yDivs_ = yDivs;
}

unsigned int MarkovSolver::getYdivs( ) const
{
	return yDivs_;
}

double MarkovSolver::getInvDy() const
{
	return invDy_;		
}


//Computes the updated state of the system. Is called from the process function.
void MarkovSolver::computeState()
{
	Vector* newState;
	Matrix *expQ = computeMatrixExponential();

	newState = vecMatMul( &state_, expQ);
	state_ = *newState;

	delete newState;
}

void MarkovSolver::innerFillupTable( MarkovRateTable *rateTable, 	
																		 vector< unsigned int > rateIndices,
																		 string rateType, unsigned int xIndex, 
																		 unsigned int yIndex )
{
	unsigned int n = rateIndices.size(), i, j;	

	for ( unsigned int k = 0; k < n; ++k )
	{
		i = ( ( rateIndices[k] / 10 ) % 10 ) - 1;
		j = ( rateIndices[k] % 10 ) - 1;

		(*Q_)[i][i] += (*Q_)[i][j];
		
		if ( rateType.compare("2D") == 0 )
			(*Q_)[i][j] = rateTable->lookup2dIndex( i, j, xIndex, yIndex );
		else if ( rateType.compare("1D") == 0 )
		{
			if ( rateTable->isRateLigandDep( i, j ) )
				(*Q_)[i][j] = rateTable->lookup1dIndex( i, j, yIndex );
			else
				(*Q_)[i][j] = rateTable->lookup1dIndex( i, j, xIndex );
		}
		else if ( rateType.compare("constant") == 0 )
			(*Q_)[i][j] = rateTable->lookup1dValue( i, j, 1.0 );

		(*Q_)[i][i] -= (*Q_)[i][j];
	}
}
																		 
void MarkovSolver::fillupTable( MarkovRateTable* rateTable )
{
	double dx = (xMax_ - xMin_) / xDivs_;  			
	double dy = (yMax_ - yMin_) / yDivs_;  			

	vector< unsigned int > listOf1dRates = rateTable->getListOf1dRates();
	vector< unsigned int > listOf2dRates = rateTable->getListOf2dRates();
	vector< unsigned int > listOfConstantRates = 
												 rateTable->getListOfConstantRates();

	//Set constant rates in the Q matrix, if any.
	innerFillupTable( rateTable, listOfConstantRates, "constant", 
										0.0, 0.0 ); 

	if ( rateTable->areAllRatesConstant() ) 
	{
		expMat_ = computeMatrixExponential();
		return;
	}

	//xIndex loops through all voltags, yIndex loops through all
	//ligand concentrations.
	double voltage = xMin_, ligandConc = yMin_;
	for ( unsigned int xIndex = 0; xIndex < xDivs_ + 1; ++xIndex )
	{
		ligandConc = yMin_;
		for( unsigned int yIndex = 0; yIndex < yDivs_ + 1; ++yIndex ) 
		{
			if ( rateTable->areAnyRates2d() )
				innerFillupTable( rateTable, listOf2dRates, "2D", xIndex, yIndex ); 

			//This is a very klutzy way of updating 1D rates as the same
			//lookup is done multiple times. But this all occurs at setup,
			//and lookups arent that slow either. This way is also easier
			//to maintain.
			if ( rateTable->areAnyRates1d() )
				innerFillupTable( rateTable, listOf1dRates, "1D", xIndex, yIndex ); 

			expMats2d_[xIndex][yIndex] = computeMatrixExponential();
			ligandConc += dy;
		}
		voltage += dx;
	}
}

Matrix* MarkovSolver::computePadeApproximant( Matrix* Q1, 
																						unsigned int degreeIndex )
{
	Matrix *expQ;
	Matrix *U, *V, *VplusU, *VminusU, *invVminusU, *Qpower;
	vector< unsigned int >* swaps = new vector< unsigned int >;
	unsigned int n = Q1->size();
	unsigned int degree = mCandidates[degreeIndex];
	double *padeCoeffs; 

	//Vector of Matrix pointers. Each entry is an even power of Q.
	vector< Matrix* > QevenPowers;

	//Selecting the right coefficient array.
	switch (degree)
	{
		case 13:
			padeCoeffs = b13;
		break;

		case 9:
			padeCoeffs = b9;
		break;

		case 7:
			padeCoeffs = b7;
		break;

		case 5:
			padeCoeffs = b5;
		break;

		case 3:
			padeCoeffs = b3;
		break;
	}

	/////////
	//Q2 = Q^2 is computed for all degrees.
	//Q4 = Q^4 = Q^2 * Q^2 is computed when degree = 5,7,9,13.
	//Q6 = Q^6 = Q^4 * Q^2 is computed when degree = 7,9,13.
	//Q8 = Q^8 = Q^4 * Q^4 is computed when degree = 7,9.
	//Note that the formula for the 13th degree approximant used here
	//is different from the one used for smaller degrees.
	////////
	switch( degree )	
	{
		case 3 : 
		case 5 :
		case 7 :
		case 9 :
			U = matAlloc( n );
			V = matAlloc( n );

			QevenPowers.push_back( Q1 );

			for( unsigned int i = 1; i < (degree + 1)/2 ; ++i )
			{
				Qpower = QevenPowers.back();
				QevenPowers.push_back( matMatMul( Qpower, Qpower ) );
			}

			//Computation of U.
			for ( int i = degree; i > 1; i -= 2 )
				matMatAdd( U, QevenPowers[i/2], 1.0, padeCoeffs[i], FIRST ); 

			//Adding b0 * I . 
			matEyeAdd( U, padeCoeffs[1], 0 );
			matMatMul( Q1, U, SECOND );

			//Computation of V.
			for ( int i = degree - 1; i > 0; i -= 2 )
				matMatAdd( V, QevenPowers[i/2], 1.0, padeCoeffs[i], FIRST );

			//Adding b1 * I
			matEyeAdd( V, padeCoeffs[0], DUMMY );

			while (!QevenPowers.empty())
			{
				delete QevenPowers.back();
				QevenPowers.pop_back();
			}
		break;

		case 13:
			Matrix *Q2, *Q4, *Q6;
			Matrix *temp;

			Q2 = matMatMul( Q1, Q1 );
			Q4 = matMatMul( Q2, Q2 );
			Q6 = matMatMul( Q4, Q2 );

			//Long and rather messy expression for U and V.
			//Refer paper mentioned in header for more details.
			//Storing the result in temporaries is a better idea as it gives us
			//control on how many temporaries are being created and also to 
			//help in memory deallocation.

			//Computation of U.
			temp = matScalShift( Q6, b13[13], 0.0 );
			matMatAdd( temp, Q4, 1.0, b13[11], FIRST );
			matMatAdd( temp, Q2, 1.0, b13[9], FIRST );
			matMatMul( Q6, temp, SECOND );
			matMatAdd( temp, Q6, 1.0, b13[7], FIRST ); 
			matMatAdd( temp, Q4, 1.0, b13[5], FIRST ); 
			matMatAdd( temp, Q2, 1.0, b13[3], FIRST ); 
			matEyeAdd( temp, b13[1], DUMMY ); 
			U = matMatMul( Q1, temp );
			delete temp;

			//Computation of V
			temp = matScalShift( Q6, b13[12], 0.0 );
			matMatAdd( temp, Q4, 1.0, b13[10], FIRST );
			matMatAdd( temp, Q2, 1.0, b13[8], FIRST );
			matMatMul( Q6, temp, SECOND );
			matMatAdd( temp, Q6, 1.0, b13[6], FIRST ); 
			matMatAdd( temp, Q4, 1.0, b13[4], FIRST ); 
			matMatAdd( temp, Q2, 1.0, b13[2], FIRST ); 
			V = matEyeAdd( temp, b13[0] ); 
			delete temp;

			delete Q2;
			delete Q4;
			delete Q6;
		break;
	}

	VplusU = matMatAdd( U, V, 1.0, 1.0 );
	VminusU = matMatAdd( U, V, -1.0, 1.0 );

	invVminusU = matAlloc( n );
	matInv( VminusU, swaps, invVminusU );
	expQ = matMatMul( invVminusU, VplusU );
	
	//Memory cleanup.
	delete U;
	delete V;
	delete VplusU;
	delete VminusU;
	delete invVminusU;
	delete swaps;

	return expQ;
}

Matrix* MarkovSolver::computeMatrixExponential()
{
	double mu;					
	unsigned int n = Q_->size();
	Matrix *expQ, *Q1;

	mu = matTrace( Q_ )/n;

	//Q1 <- Q - mu*I
	//This reduces the norm of the matrix. The idea is that a lower
	//order approximant will suffice if the norm is smaller. 
	Q1 = matEyeAdd( Q_, -mu );

	//We cycle through the first four candidate values of m. The moment the norm 
	//satisfies the theta_M bound, we choose that m and compute the Pade'
	//approximant to the exponential. We can then directly return the exponential. 
	for ( unsigned int i = 0; i < 4; ++i )
	{
		if ( matColNorm( Q1 ) < thetaM[i] )
		{
			expQ = computePadeApproximant( Q1, i );
			matScalShift( expQ, exp( mu ), 0, DUMMY );
			return expQ;
		}
	}

	//In case none of the candidates were satisfactory, we scale down the norm
	//by dividing A by 2^s until ||A|| < 1. We then use a 13th degree
	//Pade approximant.
	unsigned int s = ceil( log( matColNorm( Q1 )/thetaM[4] ) / log( 2 ) );
	if ( s > 0 )
		matScalShift( Q1, 1.0/(2 << (s - 1)), 0, DUMMY );
	expQ = computePadeApproximant( Q1, 4 );
	
	//Upto this point, the matrix stored in expQ is r13, the 13th degree
	//Pade approximant corresponding to A/2^s, not A.
	//Now we repeatedly square r13 's' times to get the exponential
	//of A.
	for ( unsigned int i = 0; i < s; ++i )
		matMatMul( expQ, expQ, FIRST );

	matScalShift( expQ, exp( mu ), 0, DUMMY );

	delete Q1;
	return expQ;
}

///////////////
//MsgDest functions
//////////////
void MarkovSolver::reinit( const Eref& e, ProcPtr p )
{
	state_ = initialState_;		
}

void MarkovSolver::process( const Eref& e, ProcPtr p )
{
	;		
}

void MarkovSolver::handleVm( double Vm )
{
	Vm_ = Vm;
}

void MarkovSolver::handleLigandConc( double ligandConc )
{
	ligandConc_ = ligandConc;
}

//Sets up the exponential lookup tables based on the rate table that is passed
//in. Initializes the whole object.
void MarkovSolver::setupTable( Id rateTableId )
{
	MarkovRateTable* rateTable = reinterpret_cast< MarkovRateTable* >(
																rateTableId.eref().data() );

	size_ = rateTable->getSize();
	setLookupParams( rateTable );

	if ( rateTable->areAnyRates2d() )
	{
		expMats2d_.resize( xDivs_ + 1 );
		for( unsigned int i = 0; i < xDivs_ + 1; ++i )
			expMats2d_[i].resize( yDivs_ + 1);
	}
	else if ( rateTable->areAnyRates1d() )
		expMats1d_.resize( xDivs_ + 1);
	else	//All rates must be constant.
		expMat_ = matAlloc( size_ );

	//Initializing Q.
	Q_ = matAlloc( size_ );		

	//Fills up the newly setup tables with exponentials.
	fillupTable( rateTable ); 
}

////////////////
//This function sets the limits of the final lookup table of matrix
//exponentials. 
//xMin_, xMax, yMin_, yMax_, xDivs_, yDivs_ are chosen such that 
//the longest inverval covering all the limits of the rate lookup 
//tables is chosen. 
//i.e. xMin_ = min( xMin of all 1D and 2D lookup tables ),
//	   xMax_ = max( xMax of all 1D and 2D lookup tables ),
//	   yMin_ = min( yMin of all 2D lookup tables ),
//	   yMax_ = max( yMax of all 2D lookup tables ),
//	   xDivs_ = min( xDivs of all 1D and 2D lookup tables ),
//		 yDivs_ = min( yDivs of all 2D lookup tables )
//
//If all the rates are constant, then all these values remain unchanged
//from the time the MarkovSolver object was constructed i.e. all are zero.
///////////////
void MarkovSolver::setLookupParams( MarkovRateTable* rateTable )
{
	if ( rateTable->areAnyRates1d() )
	{
		vector< unsigned int > listOf1dRates = rateTable->getListOf1dRates();
		double temp;
		unsigned int divs, i, j;

		for( unsigned int k = 0; k < listOf1dRates.size(); ++k )
		{
			i = ( ( listOf1dRates[k] / 10 ) % 10 ) - 1;
			j = ( listOf1dRates[k] % 10 ) - 1;

			temp = rateTable->getVtChildTable( i, j )->getMin();

			if ( !rateTable->isRateLigandDep( i, j ) )
			{
				if ( xMin_ > temp )
					xMin_ = temp;
			}
			else 
				yMin_ = temp;

			temp = rateTable->getVtChildTable( i, j )->getMax();
			if ( !rateTable->isRateLigandDep( i, j ) )
			{
				if ( xMax_ < temp )
					xMax_ = temp;
			}
			else
				yMax_ = temp;

			divs = rateTable->getVtChildTable( i, j )->getDiv();
			if ( !rateTable->isRateLigandDep( i, j ) )
			{
				if ( xDivs_ < divs )
					xDivs_ = divs;
			}
			else
				yDivs_ = divs;
		}

		invDx_ = xDivs_ / ( xMax_ - xMin_ );
	}

	if ( rateTable->areAnyRates2d() )
	{
		vector< unsigned int > listOf2dRates = rateTable->getListOf2dRates();
		double temp;
		unsigned int divs, i, j;

		for( unsigned int k = 0; k < listOf2dRates.size(); ++k )
		{
			i = ( ( listOf2dRates[k] / 10 ) % 10 ) - 1;
			j = ( listOf2dRates[k] % 10 ) - 1;

			temp = rateTable->getInt2dChildTable( i, j )->getXmin();
			if ( xMin_ > temp )
				xMin_ = temp;

			temp = rateTable->getInt2dChildTable( i, j )->getXmax();
			if ( xMax_ < temp )
				xMax_ = temp;

			temp = rateTable->getInt2dChildTable( i, j )->getYmin();
			if ( yMin_ > temp )
				yMin_ = temp;

			temp = rateTable->getInt2dChildTable( i, j )->getYmax();
			if ( yMax_ < temp )
				yMax_ = temp;

			divs = rateTable->getInt2dChildTable( i, j )->getXdivs();
			if ( xDivs_ < divs )
				xDivs_ = divs;

			divs = rateTable->getInt2dChildTable( i, j )->getYdivs();
			if ( yDivs_ < divs )
				yDivs_ = divs;
		}

		invDx_ = xDivs_ / ( xMax_ - xMin_ );
		invDy_ = yDivs_ / ( yMax_ - yMin_ );
	}
}

#ifdef DO_UNIT_TESTS
void assignMat( Matrix* A, double testMat[3][3] )
{
	for ( unsigned int i = 0; i < 3; ++i )
	{
		for ( unsigned int j = 0; j < 3; ++j )
			(*A)[i][j] = testMat[i][j];
	}
}

//In this set of tests, matrices are specially chosen so that
//we test out all degrees of the Pade approximant.
void testMarkovSolver()
{
	MarkovSolver solver;	

	Matrix *expQ;

	solver.Q_ = matAlloc( 3 );

	double testMats[5][3][3] = {
		{ //Will require 3rd degree Pade approximant.
			{ 0.003859554140797, 0.003828667792972, 0.000567545354509 },
			{ 0.000630452326710, 0.001941502594891, 0.001687045130505 },
			{ 0.003882371127042, 0.003201121875555, 0.003662942100756 }
		},
		{ //Will require 5th degree Pade approximant.
			{ 0.009032772098686, 0.000447799046538, 0.009951718245937 },
			{ 0.004122293240791, 0.005703676675533, 0.010337598714782 },
			{ 0.012352886634899, 0.004960259942209, 0.002429343859207 }
		},
		{ //Will require 7th degree Pade approximant.
			{ 0.249033679156721, 0.026663192545146, 0.193727616177876 },
			{ 0.019543882188296, 0.240474520213763, 0.204325805163358 },
			{ 0.110669567443862, 0.001158556033517, 0.217173676340877 }
		},
		{ //Will require 9th degree Pade approximant.
			{ 0.708590392291234,  0.253289557366033,  0.083402066470341 },
			{ 0.368148069351060,  0.675040384813246,  0.585189051240853 },
			{ 0.366939478800014,  0.276935085840161,  0.292304127720940 },
		},
		{ //Will require 13th degree Pade approximant.,
			{ 5.723393958255834,  2.650265678621879,  2.455089725038183 },
			{ 5.563819918184171,  5.681063207977340,  6.573010933999208 },
			{ 4.510226911355842,  3.729779121596184,  6.131599680450886 }
		}
	};

	double correctExps[5][3][3] = {
		{
			{ 1.003869332885896,  0.003840707339656,  0.000572924780299 },
			{ 0.000635569997632,  1.001947309951925,  0.001691961742250 },
			{ 0.003898019965821,  0.003217566115560,  1.003673477658833 }
		},
		{
			{ 1.009136553319587,  0.000475947724581,  0.010011555682222 },
			{ 0.004217120071231,  1.005746704105642,  0.010400661735110 },
			{ 0.012434554824033,  0.004983402186283,  1.002519822649746 }
		},
		{
			{ 1.296879503336410,  0.034325768324765,  0.248960074106229 },
			{ 0.039392602584525,  1.272463533413523,  0.260228538022164 },
			{ 0.140263068698347,  0.003332731847933,  1.256323839764616 }
		},
		{
			{ 2.174221102665550,  0.550846463313377,  0.279203836454105 },
			{ 0.963674962388503,  2.222317715620410,  1.033020817699738 },
			{ 0.733257221615105,  0.559435366776953,  1.508376826849517 }
		},
		{
			{ 3.274163243250245e5,  2.405867301580962e5,  3.034390382218154e5 },
			{ 5.886803379935408e5,  4.325844111569120e5,  5.456065763194024e5 },
			{ 4.671930521670584e5,  3.433084310645007e5,  4.330101744194682e5 }
		}
	};

	double correctColumnNorms[5] = {
	 		1.009005583407142,
			1.025788228214852,
			1.765512451893010,
			3.871153286669158,
			1.383289714485624e+06
	};

	for ( unsigned int i = 0; i < 5; ++i )
	{
		assignMat( solver.Q_, testMats[i] );
		expQ = solver.computeMatrixExponential();	
		assert( doubleEq( matColNorm( expQ ), correctColumnNorms[i] ) ); 

		//Comparing termwise just to be doubly sure.
		for( unsigned int j = 0; j < 3; ++j )
		{
			for( unsigned int k = 0; k < 3; ++k )
				assert( doubleEq( (*expQ)[j][k], correctExps[i][j][k] ) );
		}

		delete expQ;
	}

	cout << "." << flush;
}

 #endif
