#include "header.h"
#include "Job.h"
#include "JobWrapper.h"
#include "ClockJob.h"
#include "ClockJobWrapper.h"


Finfo* ClockJobWrapper::fieldArray_[] =
{
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	new ValueFinfo< double >(
		"runTime", &ClockJobWrapper::getRuntime, 
		&ClockJobWrapper::setRuntime, "double" ),
	new ReadOnlyValueFinfo< double >(
		"currentTime", &ClockJobWrapper::getCurrentTime, "double" ),
	new ValueFinfo< int >(
		"nSteps", &ClockJobWrapper::getNSteps, 
		&ClockJobWrapper::setNSteps, "int" ),
	new ReadOnlyValueFinfo< int >(
		"currentStep", &ClockJobWrapper::getCurrentStep, "int" ),
///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
	new NSrc1Finfo< ProcInfo >(
		"processOut", &ClockJobWrapper::getProcessSrc, 
		"", 1 ),
	new NSrc0Finfo(
		"reschedOut", &ClockJobWrapper::getReschedSrc, 
		"reschedIn, resetIn", 1 ),
	new NSrc0Finfo(
		"reinitOut", &ClockJobWrapper::getReinitSrc, 
		"reinitIn, resetIn", 1 ),
	new NSrc0Finfo(
		"finishedOut", &ClockJobWrapper::getFinishedSrc, "" ),
///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
	new Dest1Finfo< ProcInfo >(
		"startIn", &ClockJobWrapper::startFunc,
		&ClockJobWrapper::getStartInConn, "" ),
	new Dest2Finfo< ProcInfo, int >(
		"stepIn", &ClockJobWrapper::stepFunc,
		&ClockJobWrapper::getStepInConn, "" ),
	new Dest2Finfo< double, Conn* >(
		"dtIn", &ClockJobWrapper::dtFunc,
		&ClockJobWrapper::getClockConn, "", 1 ),
	new Dest0Finfo(
		"reinitIn", &ClockJobWrapper::reinitFunc,
		&ClockJobWrapper::getReinitInConn, "reinitOut" ),
	new Dest0Finfo(
		"reschedIn", &ClockJobWrapper::reschedFunc,
		&ClockJobWrapper::getReschedInConn, "reschedOut" ),
	new Dest0Finfo(
		"resetIn", &ClockJobWrapper::resetFunc,
		&ClockJobWrapper::getResetInConn, "reschedOut, reinitOut" ),
///////////////////////////////////////////////////////
// Synapse definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Shared definitions
///////////////////////////////////////////////////////
	new SharedFinfo(
		"clock", &ClockJobWrapper::getClockConn,
		"processOut, reinitOut, reschedOut, dtIn" ),
};

const Cinfo ClockJobWrapper::cinfo_(
	"ClockJob",
	"Upinder S. Bhalla, Nov 2005, NCBS",
	"ClockJob: ClockJob class. Handles sequencing of operations in simulations",
	"Neutral",
	ClockJobWrapper::fieldArray_,
	sizeof(ClockJobWrapper::fieldArray_)/sizeof(Finfo *),
	&ClockJobWrapper::create
);

///////////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Synapse function definitions
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Connection function definitions
///////////////////////////////////////////////////////
Element* startCJInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, startInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* stepCJInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, stepInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reinitCJInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, reinitInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* reschedCJInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, reschedInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

Element* resetCJInConnLookup( const Conn* c )
{
	static const unsigned long OFFSET =
		FIELD_OFFSET ( ClockJobWrapper, resetInConn_ );
	return reinterpret_cast< ClockJobWrapper* >( ( unsigned long )c - OFFSET );
}

////////////////////////////////////////////////////////////////
// Local functions for ClockJob
////////////////////////////////////////////////////////////////
void ClockJobWrapper::startFuncLocal( ProcInfo info )
{
	cout << "starting run for " << runTime_ << " sec.\n";
	info->currTime_ = currentTime_;
	if ( tickSrc_ )
		tickSrc_->start( info, currentTime_ + runTime_ );
	currentTime_ = info->currTime_;
}

void ClockJobWrapper::stepFuncLocal( ProcInfo info, int nsteps )
{
	cout << "starting run for " << nsteps << " steps.\n";
	info->currTime_ = currentTime_;
	if ( tickSrc_ ) {
		runTime_ = nsteps * tickSrc_->dt();
		tickSrc_->start( info, currentTime_ + runTime_ );
	}
	currentTime_ = info->currTime_;
}

void ClockJobWrapper::sortTicks( )
{
	// Bubble sort till done. There is only one changed entry, so not
	// too bad.
	ClockTickMsgSrc** i;
	ClockTickMsgSrc** j = &tickSrc_;
	bool swapping = 1;
	while ( swapping ) {
		swapping = 0;
		for ( i = (*j)->next(); *i != 0; i = (*i)->next() ) {
			if ( **i < **j ) {
				(*i)->swap( j );
				i = j;
				swapping = 1;
				break;
			} else {
				j = i;
			}
		}
	}
	tickSrc_->updateNextClockTime();
}

// 1. Locate the affected ClockTickMsgSrc.
// 2. Set up the altered dt, making a new ClockTickMsgSrc if needed
// 3. Sort
// 4. Merge ClockTickMsgSrcs if possible
void ClockJobWrapper::dtFuncLocal( double dt, Conn* tick )
{
	if ( tickSrc_ == 0 )
		return;

	// Here do 1 and 2.
	tickSrc_->updateDt( dt, tick );

	sortTicks(); // Here we do number 3.

	// Here we do  4: Merge identical ones.
	ClockTickMsgSrc** i;
	ClockTickMsgSrc** j = &tickSrc_;
	for ( i = (*j)->next(); *i != 0; i = (*i)->next() ) {
		if ( **i == **j ) {
		//	ClockTickMsgSrc* k = *i;
			( *j )->merge( *i );
			i = j;
		}
		/*
		else if ( **i < **j ) {
			(*i)->swap( j );
			i = j;
		}
		*/
		j = i;
	}
	/*
	// Bubble sort till done. There is only one changed entry, so not
	// too bad.
	j = &tickSrc_;
	bool swapping = 1;
	while ( swapping ) {
		swapping = 0;
		for ( i = (*j)->next(); *i != 0; i = (*i)->next() ) {
			if ( **i < **j ) {
				(*i)->swap( j );
				i = j;
				swapping = 1;
				break;
			} else {
				j = i;
			}
		}
	}
	*/
}

void ClockJobWrapper::reinitFuncLocal( )
{
	currentTime_ = 0.0;
	currentStep_ = 0;
	if ( tickSrc_ )
		tickSrc_->reinit();
}

void ClockJobWrapper::reschedFuncLocal( )
{
	// Clear out old tickSrc_
	if ( tickSrc_ )
		delete ( tickSrc_ );
	tickSrc_ = 0;

	// Build list of child ClockTicks.
	vector< Field > f;
	Field kids = field( "child_out" );
	kids->dest( f, this );

	// Scan list, get dts and rfuncs for proc, reinit and resched
	vector< Field >::iterator i;
	vector< ClockTickMsgSrc > ticks;
	for ( i = f.begin(); i != f.end(); i++ )
	 	ticks.push_back( ClockTickMsgSrc( this, i->getElement() ) );

	// Build list of ClockTickMsgSrc according to dts. May need
	// additional info in case we have specific sequence for same dt.
	// This could be a field in ClockTick specifying which it follows.
	// Check if there is an algorithm for sorting pointers.
	sort( ticks.begin(), ticks.end() );
	vector< ClockTickMsgSrc >::iterator j;
	// vector< ClockTickMsgSrc >::iterator prev = ticks.end();
	ClockTickMsgSrc** currTick = &tickSrc_;
	ClockTickMsgSrc* prev = tickSrc_;

	// Build connections
	for ( j = ticks.begin(); j != ticks.end(); j++ ) {
		if ( j == ticks.begin() || !( *prev == *j ) ) {
			// Make a new tick and connect to internal target.
			*currTick = new ClockTickMsgSrc( *j );
			( *currTick )->connect( **currTick );
			prev = *currTick;
			currTick = (*currTick)->next();
		} else { // Connect last tick to target of j.
			prev->connect( *j ); // Connect the target of j
		}
	}

	// Go through new list, call the resched for each.
	if ( tickSrc_ ) {
		tickSrc_->resched( );
	}
}

////////////////////////////g///////////////////////////////////
// Member functions for ClockTickMsgSrc
////////////////////////////////////////////////////////////////

ClockTickMsgSrc::ClockTickMsgSrc( Element* e, Element* target )
	: dt_( 1.0 ), nextTime_( 0.0 ), nextClockTime_( 0.0 ),
		conn_( e ), next_( 0 ), target_( target )
{
 	if ( Ftype1< double >::get( target, "dt", dt_ ) ) {
 		if ( Ftype1< double >::get( target, "stage", stage_ ) ) {
			// Hacks all, to set up the recvFuncs in either direction.
			procFunc_ = target->field( "processIn" )->recvFunc();
			reinitFunc_ = target->field( "reinitIn" )->recvFunc();
			reschedFunc_ = target->field( "reschedIn" )->recvFunc();
			RecvFunc rf = e->field( "dtIn" )->recvFunc();
			target->field( "dtOut" )->addRecvFunc( target, rf, 0 );
			return;
		}
	}
	cerr << "Error: Failed to initialize ClockTickMsgSrc\n";
}

ClockTickMsgSrc::~ClockTickMsgSrc( )
{
	// Disconnect!! Actually the destruction of the Conn should do it.
	if ( next_ )
		delete next_;
}

void ClockTickMsgSrc::connect( ClockTickMsgSrc& dest )
{
	Element* target = dest.target_;
	Conn* c = target->field( "clock" )->inConn( target );
	if ( c )
		conn_.connect( c, 0 );
	else 
		cerr << "Warning: ClockTickMsgSrc::connect() failed for " <<
			target->path() << "\n";
}

void ClockTickMsgSrc::reinit( )
{
	for_each( conn_.begin(), conn_.end(), reinitFunc_ );
	nextTime_ = 0;
	if ( next_ )
		next_->reinit( );
}

void ClockTickMsgSrc::resched( )
{
	for_each( conn_.begin(), conn_.end(), reschedFunc_ );
	if ( next_ )
		next_->resched( );
}

void ClockTickMsgSrc::setProcInfo( ProcInfo info )
{
	op_ = Op1< ProcInfo >( info, 
		reinterpret_cast< void ( * )( Conn*, ProcInfo )> ( procFunc_ ));
	if ( next_ )
		next_->setProcInfo( info );
}

//    Here is the func that scans through child clockticks
// It can be stopped if info.halt == 1. It can resume without
// a hiccup at any point.
// It stops as soon as any clock exceeds maxTime.
void ClockTickMsgSrc::start( ProcInfo info, double maxTime )
{
	static double NEARLY_ONE = 0.999999999999;
	static double JUST_OVER_ONE = 1.000000000001;
	double endTime;
	setProcInfo( info );
	maxTime = maxTime * NEARLY_ONE;

	while ( info->currTime_ < maxTime ) {
		endTime = maxTime + dt_;
		if ( next_ && endTime > nextClockTime_ )
			endTime = nextClockTime_ * JUST_OVER_ONE;

		info->dt_ = dt_;
		while ( nextTime_ <= endTime ) {
			info->currTime_ = nextTime_;
			for_each( conn_.begin(), conn_.end(), op_ );
			nextTime_ += dt_;
		}

		if ( next_ ) {
			while ( nextClockTime_ < nextTime_ )
				nextClockTime_ = 
					next_->incrementClock( info, nextTime_ );
		}
	}
}

// Increment whichever clock is due for updating. Cascades to multiple
// clocks if possible.
// Return the smallest nextt of this and further clocks
// Assume dt of successive clocks is in increasing order.
// Assume that prevClockTime exceeds nextTime_: this func wouldn't be
// called otherwise.
double ClockTickMsgSrc::incrementClock( 
	ProcInfo info, double prevClockTime )
{
	if ( next_ ) {
		if ( nextTime_ <= nextClockTime_ ) {
			info->currTime_ = nextTime_;
			info->dt_ = dt_;
			for_each( conn_.begin(), conn_.end(), op_ );
			nextTime_ += dt_;
		}
		if ( nextTime_ > nextClockTime_ && 
			prevClockTime > nextClockTime_ ) {
			nextClockTime_ = 
				next_->incrementClock( info, prevClockTime );
		}
		if ( nextClockTime_ < nextTime_ )
			return nextClockTime_;
	} else {
		info->currTime_ = nextTime_;
		info->dt_ = dt_;
		for_each( conn_.begin(), conn_.end(), op_ );
		nextTime_ += dt_;
	}
	return nextTime_;
}


ClockTickMsgSrc* ClockTickMsgSrc::findSrcOf( const Conn* tick )
{
	unsigned long j = conn_.nTargets();
	for ( unsigned long i = 0; i < j; i++ )
		if ( conn_.target( i ) == tick )
			return this;
	if ( next_ )
		return next_->findSrcOf( tick );
	return 0;
}

void ClockTickMsgSrc::updateDt( double newdt, Conn* tick )
{
	// 1. Locate the affected ClockTickMsgSrc.
	ClockTickMsgSrc* ct = findSrcOf( tick );
	if ( ct == 0 ) {
		cerr << "Error: ClockTickMsgSrc::updateDt: Failed to find msg src matching " << tick->parent()->path() << "\n";
		return;
	}

	// 2. Set up the altered dt, making a new ClockTickMsgSrc if needed
	if ( ct->conn_.nTargets() == 1 ) { // Just reassign current ct
		ct->nextTime_ += newdt - ct->dt_;
		ct->dt_ = newdt;
		//ct->stage_ = stage;
	} else { // spawn off a new ClockTickMsgSrc with the new dt.
		ClockTickMsgSrc* nt = new ClockTickMsgSrc( *ct );
		ct->conn_.disconnect( tick );
		nt->conn_.innerDisconnectAll();
		nt->conn_.connect( tick, 0, 0 );
		nt->nextTime_ += newdt - ct->dt_;
		nt->dt_ = newdt;
		//nt->stage_ = stage;
		ct->next_ = nt; // nt->next is already set to ct->next.
	}
}

// Swaps the order of the ClockTickMsgSrc in their linked list
// this is the second
// other is the first, so (*other)->next_ == this
void ClockTickMsgSrc::swap( ClockTickMsgSrc** other )
{
	ClockTickMsgSrc* temp = next_;
	next_ = *other;
	( *other )->next_ = temp;
	*other = this;
}

// Current ClockTickMsgSrc absorbs its successor named other.
void ClockTickMsgSrc::merge( ClockTickMsgSrc* other )
{
	if ( next_ != other ) {
		cerr << "Error: ClockTickMsgSrc::merge: ordering wrong\n";
		return;
	}
	next_ = other->next_;
	other->next_ = 0;
	unsigned long i;
	vector< Conn* > connVec;
	other->conn_.listTargets( connVec );
	other->conn_.disconnectAll( );

	for ( i = 0; i < connVec.size(); i++ )
		conn_.connect( connVec[ i ], 0 );

	delete other;
}

void ClockTickMsgSrc::updateNextClockTime( )
{
	if ( next_ ) {
		nextClockTime_ = next_->nextTime_;
		next_->updateNextClockTime();
	} else {
		nextClockTime_ = 0;
	}
}
