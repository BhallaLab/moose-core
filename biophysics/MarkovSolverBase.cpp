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

#include "MarkovSolverBase.h"

SrcFinfo1< Vector >* stateOut()
{
	static SrcFinfo1< Vector > stateOut("stateOut",
		"Sends updated state to the MarkovChannel class."
		);
	return &stateOut;
}

const Cinfo* MarkovSolverBase::initCinfo()
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
			"This message couples the MarkovSolverBase to the Compartment. The "
			"compartment needs Vm in order to look up the correct matrix "
			"exponential for computing the state.",
			channelShared, sizeof( channelShared ) / sizeof( Finfo* ) 
			);

	//////////////////////
	//DestFinfos
	//////////////////////
	
	static DestFinfo process(	"process",
			"Handles process call",
			new ProcOpFunc< MarkovSolverBase >( &MarkovSolverBase::process ) ); 

	static DestFinfo reinit( "reinit", 
			"Handles reinit call",
			new ProcOpFunc< MarkovSolverBase >( &MarkovSolverBase::reinit ) );

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
			new OpFunc1< MarkovSolverBase, double >(&MarkovSolverBase::handleLigandConc) 
			);

	static DestFinfo setuptable("setuptable",
			"Setups the table of matrix exponentials associated with the"
			" solver object.",
			new OpFunc1< MarkovSolverBase, Id >(&MarkovSolverBase::setupTable) 
			);

	//////////////////////
	//*ValueFinfos
	/////////////////////
	
	static ReadOnlyValueFinfo< MarkovSolverBase, Matrix > Q("Q",
			"Instantaneous rate matrix.",
			&MarkovSolverBase::getQ
			);

	static ReadOnlyValueFinfo< MarkovSolverBase, Vector > state("state",
			"Current state of the channel.",
			&MarkovSolverBase::getState 
			);

	static ValueFinfo< MarkovSolverBase, Vector > initialstate("initialstate",
			"Initial state of the channel.",
			&MarkovSolverBase::setInitialState,
			&MarkovSolverBase::getInitialState 
			);

	static ValueFinfo< MarkovSolverBase, double > xmin( "xmin",
		"Minimum value for x axis of lookup table",
			&MarkovSolverBase::setXmin,
			&MarkovSolverBase::getXmin
		);
	static ValueFinfo< MarkovSolverBase, double > xmax( "xmax",
		"Maximum value for x axis of lookup table",
			&MarkovSolverBase::setXmax,
			&MarkovSolverBase::getXmax
		);
	static ValueFinfo< MarkovSolverBase, unsigned int > xdivs( "xdivs",
		"# of divisions on x axis of lookup table",
			&MarkovSolverBase::setXdivs,
			&MarkovSolverBase::getXdivs
		);
	static ReadOnlyValueFinfo< MarkovSolverBase, double > invdx( "invdx",
		"Reciprocal of increment on x axis of lookup table",
			&MarkovSolverBase::getInvDx
		);
	static ValueFinfo< MarkovSolverBase, double > ymin( "ymin",
		"Minimum value for y axis of lookup table",
			&MarkovSolverBase::setYmin,
			&MarkovSolverBase::getYmin
		);
	static ValueFinfo< MarkovSolverBase, double > ymax( "ymax",
		"Maximum value for y axis of lookup table",
			&MarkovSolverBase::setYmax,
			&MarkovSolverBase::getYmax
		);
	static ValueFinfo< MarkovSolverBase, unsigned int > ydivs( "ydivs",
		"# of divisions on y axis of lookup table",
			&MarkovSolverBase::setYdivs,
			&MarkovSolverBase::getYdivs
		);
	static ReadOnlyValueFinfo< MarkovSolverBase, double > invdy( "invdy",
		"Reciprocal of increment on y axis of lookup table",
			&MarkovSolverBase::getInvDy
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

	static Cinfo markovSolverBaseCinfo(
			"MarkovSolverBase",			
			Neutral::initCinfo(),
			markovSolverFinfos,
			sizeof( markovSolverFinfos ) / sizeof( Finfo* ),
			new Dinfo< MarkovSolverBase > 
			);

	return &markovSolverBaseCinfo;
}

static const Cinfo* markovSolverBaseCinfo = MarkovSolverBase::initCinfo();

MarkovSolverBase::MarkovSolverBase() : Q_(0), expMats1d_(0), expMat_(0), 
	expMats2d_(0), xMin_(DBL_MAX), xMax_(DBL_MIN), xDivs_(0u), 
	yMin_(DBL_MAX), yMax_(DBL_MIN), yDivs_(0u), size_(0u), Vm_(0),
 	ligandConc_(0)
{
	;
}

MarkovSolverBase::~MarkovSolverBase()
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
Matrix MarkovSolverBase::getQ() const
{
	return *Q_;
}

Vector MarkovSolverBase::getState() const
{
	return state_;
}

Vector MarkovSolverBase::getInitialState() const
{
	return initialState_;
}

void MarkovSolverBase::setInitialState( Vector state )
{
	state_ = state;
}

void MarkovSolverBase::setXmin( double xMin )
{
	xMin_ = xMin;
}

double MarkovSolverBase::getXmin() const
{
	return xMin_;
}

void MarkovSolverBase::setXmax( double xMax ) 
{
	xMax_ = xMax;	
}

double MarkovSolverBase::getXmax() const
{
	return xMax_;
}

void MarkovSolverBase::setXdivs( unsigned int xDivs )
{
	xDivs_ = xDivs;
}

unsigned int MarkovSolverBase::getXdivs( ) const {
	return xDivs_;
}

double MarkovSolverBase::getInvDx() const {
	return invDx_;
}

void MarkovSolverBase::setYmin( double yMin )
{
	yMin_ = yMin;
}

double MarkovSolverBase::getYmin() const
{
	return yMin_;
}

void MarkovSolverBase::setYmax( double yMax )
{
	yMax_ = yMax;
}

double MarkovSolverBase::getYmax() const
{
	return yMax_;
}

void MarkovSolverBase::setYdivs( unsigned int yDivs )
{
	yDivs_ = yDivs;
}

unsigned int MarkovSolverBase::getYdivs( ) const
{
	return yDivs_;
}

double MarkovSolverBase::getInvDy() const
{
	return invDy_;		
}

//Computes the updated state of the system. Is called from the process function.
void MarkovSolverBase::computeState()
{
	Vector* newState;
	Matrix *expQ = computeMatrixExponential();

	newState = vecMatMul( &state_, expQ);
	state_ = *newState;

	delete newState;
}

void MarkovSolverBase::innerFillupTable( MarkovRateTable *rateTable, 	
																		 vector< unsigned int > rateIndices,
																		 string rateType, 
																		 unsigned int xIndex, 
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
																		 
void MarkovSolverBase::fillupTable( MarkovRateTable* rateTable )
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

Matrix* MarkovSolverBase::computeMatrixExponential() 
{
	return 0;
}

///////////////
//MsgDest functions
//////////////

void MarkovSolverBase::reinit( const Eref& e, ProcPtr p )
{
	state_ = initialState_;		
}

void MarkovSolverBase::process( const Eref& e, ProcPtr p )
{
	;		
}

void MarkovSolverBase::handleVm( double Vm )
{
	Vm_ = Vm;
}

void MarkovSolverBase::handleLigandConc( double ligandConc )
{
	ligandConc_ = ligandConc;
}

//Sets up the exponential lookup tables based on the rate table that is passed
//in. Initializes the whole object.
void MarkovSolverBase::setupTable( Id rateTableId )
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
//from the time the MarkovSolverBase object was constructed i.e. all are zero.
///////////////
void MarkovSolverBase::setLookupParams( MarkovRateTable* rateTable )
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
