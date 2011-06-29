/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "MatrixOps.h"
#include "MarkovSolver.h"

SrcFinfo1< Vector >* stateOut()
{
	static SrcFinfo1< Vector > stateOut("stateOut",
		"Sends updated state to the MarkovChannel class."
		);
	return &stateOut;
}

MarkovSolver::MarkovSolver() : Q_(0)
{
	;
}

MarkovSolver::~MarkovSolver()
{
	if ( Q_ != 0 )
		delete Q_;
}

Matrix MarkovSolver::getQ() const
{
	return *Q_;
}

void MarkovSolver::computeState()
{
	Vector* newState;
	Matrix *expQ = computeMatrixExponential();

	newState = vecMatMul( state_, *expQ);
	state_ = *newState;
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
			//control on how many temporaries are being destroyed and also to 
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
	Q1 = matEyeAdd( Q_, -mu );

	//We cycle through the first four candidate values of m. The moment the norm is
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

void MarkovSolver::handleInstRates( Matrix Q )
{
	*Q_ = Q;
}

void MarkovSolver::setInitState( Vector state )
{
	state_ = state;
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
