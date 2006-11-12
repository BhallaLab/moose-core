/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "MultiSite.h"
#include "MultiSiteWrapper.h"


Finfo* MultiSiteWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"nTotal", &MultiSiteWrapper::getNTotal, 
		&MultiSiteWrapper::setNTotal, "double" ),
	new ArrayFinfo< int >(
		"states", &MultiSiteWrapper::getStates, 
		&MultiSiteWrapper::setStates, "int" ),
	new ArrayFinfo< double >(
		"occupancy", &MultiSiteWrapper::getOccupancy, 
		&MultiSiteWrapper::setOccupancy, "double" ),
	new ArrayFinfo< double >(
		"rates", &MultiSiteWrapper::getRates, 
		&MultiSiteWrapper::setRates, "double" ),
///////////////////////////////////////////////////////
// EvalField definitions
///////////////////////////////////////////////////////
	new ValueFinfo< int >(
		"nSites", &MultiSiteWrapper::getNSites, 
		&MultiSiteWrapper::setNSites, "int" ),
	new ValueFinfo< int >(
		"nStates", &MultiSiteWrapper::getNStates, 
		&MultiSiteWrapper::setNStates, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< double >(
		"scaleOut", &MultiSiteWrapper::getScaleSrc, 
		"processIn" ),
	new SingleSrc3Finfo< const vector< int >*, const vector< double >*, int >(
		"solveOut", &MultiSiteWrapper::getSolveSrc, 
		"", 1 ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< double >(
		"siteIn", &MultiSiteWrapper::siteFunc,
		&MultiSiteWrapper::getSiteInConn, "" ),
	new Dest0Finfo(
		"reinitIn", &MultiSiteWrapper::reinitFunc,
		&MultiSiteWrapper::getProcessConn, "", 1 ),
	new Dest1Finfo< ProcInfo >(
		"processIn", &MultiSiteWrapper::processFunc,
		&MultiSiteWrapper::getProcessConn, "scaleOut", 1 ),
	new Dest0Finfo(
		"solveIn", &MultiSiteWrapper::solveFunc,
		&MultiSiteWrapper::getSolveConn, "", 1 ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"process", &MultiSiteWrapper::getProcessConn,
		"processIn, reinitIn" ),
	new SharedFinfo(
		"solve", &MultiSiteWrapper::getSolveConn,
		"processIn, reinitIn, solveIn, solveOut" ),
};

const Cinfo MultiSiteWrapper::cinfo_(
	"MultiSite",
	"Upinder S. Bhalla, 2006, NCBS",
	"MultiSite: A multi-site molecule or complex. Manages a number of sites\nwhich can be either bound or unbound (2 states only). Each\nof these is reprepresented as a regular molecule pool.\nConsider a MultiSite molecule MS. It has 3 sites, S1, S2 and S3.\nReaction rate does not depend on S1 at all.\nWe have basal rate for S2 and S3 both bound\nWe have a high rate if S2 is empty and S3 is bound\nWe have a low rate if S2 is bound and S3 is in any state.\nS1  S2  S3\n|   |   |\n|   |   |\n|   |   |\nV   V   V\nMultiSite\nf1  f2  f3  : Compute fraction of molecules in each state.\n-1  0   1   : State array. ----> occupancy 1\n-1  1  -1   : State array. ----> occupancy 2.\nNote that occupancy zero is calculated as what is left\nover from the others.\nSuppose we have total MS = 3. Let\n[S1] = 1, [S2] = 1. [S3] = 2. Then,\nf1_1 = 1 (no effect). f2_1 = 1/3, f3_1 = 1/2\nOccupancy 0 = 1 * 2/3 * 1/2 = 1/3           \nOccupancy 1 = 1 * 2/3 * 1/2 = 1/3\nOccupancy 2 = 1 * 1/3 * 1   = 1/3\nSo we end up with equal # of molecules in each state:\n1/3, 1/3, 1/3.\nSay reaction is conversion of S0 to S0*\nSo total rate should be\nOccupancy0 * kf * (basal = 1) +\nOccupancy1 * kf * rate1 +\nOccupancy2 * kf * rate2\nOr,\nrate = substrate concs * kf * scale_factor.\nTo implement all these calculations, the MultiSite uses \n3 arrays:\nstates: Array of ints that defines each state.\nEach set has as many ints as there are sites.\nEach int can take one of three values:\n0: Site must be unoccupied\n1: Site must be occupied\n-1: Site does not matter.\nsize is # of states * # of sites.\nThe values in states are set by user.\noccupancy: Array of doubles holding proportion of the Multisite\nin the specified state. Size = # of states.\nOccupancy zero is always calculated as (1 - the rest).\nThe values in this array are calculated by object.\nrates: Array of doubles holding the new rates for each\nstate. Assuming occupancy of state were 1, the reaction\nrate is scaled by this new rate. \nThe rate values are set by user.\nAfter all this, a single scale factor is computed, which\nis sigma( occupancy_i * rate_i ). This scale factor is\nsent out to the target reaction(s).",
	"Neutral",
	MultiSiteWrapper::fieldArray_,
	sizeof(MultiSiteWrapper::fieldArray_)/sizeof(Finfo *),
	&MultiSiteWrapper::create
);

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void MultiSiteWrapper::setStates(
	Element* e , unsigned long index, int value )
{
	MultiSiteWrapper* f = static_cast< MultiSiteWrapper* >( e );
	if ( f->states_.size() > index )
		f->states_[ index ] = value;
}

int MultiSiteWrapper::getStates(
	const Element* e , unsigned long index )
{
	const MultiSiteWrapper* f = static_cast< const MultiSiteWrapper* >( e );
	if ( f->states_.size() > index )
		return f->states_[ index ];
	return f->states_[ 0 ];
}

void MultiSiteWrapper::setOccupancy(
	Element* e , unsigned long index, double value )
{
	MultiSiteWrapper* f = static_cast< MultiSiteWrapper* >( e );
	if ( f->occupancy_.size() > index )
		f->occupancy_[ index ] = value;
}

double MultiSiteWrapper::getOccupancy(
	const Element* e , unsigned long index )
{
	const MultiSiteWrapper* f = static_cast< const MultiSiteWrapper* >( e );
	if ( f->occupancy_.size() > index )
		return f->occupancy_[ index ];
	return f->occupancy_[ 0 ];
}

void MultiSiteWrapper::setRates(
	Element* e , unsigned long index, double value )
{
	MultiSiteWrapper* f = static_cast< MultiSiteWrapper* >( e );
	if ( f->rates_.size() > index )
		f->rates_[ index ] = value;
}

double MultiSiteWrapper::getRates(
	const Element* e , unsigned long index )
{
	const MultiSiteWrapper* f = static_cast< const MultiSiteWrapper* >( e );
	if ( f->rates_.size() > index )
		return f->rates_[ index ];
	return f->rates_[ 0 ];
}


///////////////////////////////////////////////////
// EvalField function definitions
///////////////////////////////////////////////////

int MultiSiteWrapper::localGetNSites() const
{
			return fraction_.size();
}
void MultiSiteWrapper::localSetNSites( int value) {
			if ( value > 0 ) {
				vector< unsigned long > segments( 1, value );
				states_.resize( value * occupancy_.size() );
				fraction_.resize( value );
				siteInConn_.resize( segments );
			}
}
int MultiSiteWrapper::localGetNStates() const
{
			return occupancy_.size();
}
void MultiSiteWrapper::localSetNStates( int value) {
			if ( value > 0 ) {
				occupancy_.resize( value );
				rates_.resize( value );
				states_.resize( fraction_.size() * value );
			}
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void MultiSiteWrapper::reinitFuncLocal(  )
{
			if ( fraction_.size() != siteInConn_.nTargets() ) {
				cout << "Warning: MultiSite::reinit: nSites != number of site messages (" <<
				fraction_.size() << " != " <<
				siteInConn_.nTargets() << endl;
			}
}
void MultiSiteWrapper::processFuncLocal( ProcInfo info )
{
			double totalOccupancy = 0.0;
			int nStates = occupancy_.size();
			int nSites = fraction_.size();
			for (int i = 0; i < nStates; i++ ) {
				double scale = 1.0;
				for (int j = 0; j < nSites; j++ ) {
					int state =  states_[j + nSites * i];
					if ( state == 0 ) {
						scale *= 1.0 - fraction_[j];
					} else if ( state == 1 ) {
						scale *= fraction_[j];
					}
				}
				occupancy_[i] = scale;
				totalOccupancy += scale;
			}
			double temp = (1.0 - totalOccupancy);
			for (int i = 0; i < nStates; i++ )
				temp += occupancy_[i] * rates_[i];
			scaleSrc_.send( temp );
}
///////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////
Element* processConnMultiSiteLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MultiSiteWrapper, processConn_ );
	return reinterpret_cast< MultiSiteWrapper* >( ( unsigned long )c - OFFSET );
}

Element* solveConnMultiSiteLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( MultiSiteWrapper, solveConn_ );
	return reinterpret_cast< MultiSiteWrapper* >( ( unsigned long )c - OFFSET );
}

///////////////////////////////////////////////////
// Other function definitions
///////////////////////////////////////////////////
bool MultiSiteWrapper::isSolved() const
{
	return ( solveSrc_.targetFunc(0) && 
		solveSrc_.targetFunc(0) != dummyFunc0 );
}
void MultiSiteWrapper::solverUpdate( const Finfo* f, SolverOp s ) const
{
	if ( solveSrc_.targetFunc(0) && 
		solveSrc_.targetFunc(0) != dummyFunc0 ) {
		if ( s == SOLVER_SET ) {
			if ( f->name() == "states" || f->name() == "rates" ||
				f->name() == "nSites" || f->name() == "nStates" )
				solveSrc_.send( &states_, &rates_, SOLVER_SET );
		} else if ( s == SOLVER_GET ) {
		}
	}
}
