#include <queue>
#include "header.h"
#include "IntFire.h"

const Slot spikeSlot = 0;

IntFire::IntFire( double thresh, double tau )
	: Vm_( 0.0 ), thresh_( thresh ), tau_( tau ), X_( 0.0 ), Y_( 0.0 )
{
	;
}

Finfo** IntFire::initClassInfo()
{
	static Finfo* intFireFinfos[] = {
		new Finfo( async1< IntFire, double, &IntFire::setVm > ),
		new Finfo( async1< IntFire, double, &IntFire::setThresh > ),
		new Finfo( async1< IntFire, double, &IntFire::setTau > ),
	};

	return intFireFinfos;
}

void IntFire::process( const ProcInfo* p, Eref e )
{
	while ( !pendingEvents_.empty() &&
		pendingEvents_.top().delay <= p->currTime ) {
			Vm_ += pendingEvents_.top().weight;
			pendingEvents_.pop();
	}
	if ( Vm_ > thresh_ ) {
		e.sendSpike( spikeSlot, p->currTime );
		Vm_ = -1.0e-7;
	} else {
		Vm_ *= ( 1.0 - p->dt / tau_ );
	}


/* This is what we would do for a conductance  channel.
	X_ = activation * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	*/
	 
/*
	unsigned int synSize = sizeof( SynInfo );
	for( char* i = e.processQ.begin(); i != e.processQ.end(); i += synSize )
	{
		SynInfo* si = static_cast< SynInfo* >( i );
		insertQ( si );
	}
	
	SynInfo* si = processQ.top();
	double current = 0.0;
	while ( si->time < p->time && si != processQ.end() ) {
		current += si->weight;
	}

	v_ += current * Gm_ + Em_ - tau_ * v_;
	if ( v_ > vThresh ) {
		v_ = Em_;
		sendWithId< double >( e, spikeSlot, p->t );
	}
*/
}

/**
 * Inserts an event into the pendingEvents queue for spikes.
 */
void IntFire::addSpike( unsigned int id, double time )
{
	assert( id < synapses_.size() );
	SynInfo s( synapses_[ id ], time );
	pendingEvents_.push( s );
}

void IntFire::reinit( Eref e )
{
	// pendingEvents_.resize( 0 );
	while( !pendingEvents_.empty() )
		pendingEvents_.pop();
	Vm_ = 0.0;
}

/*
void IntFire::clearQ( Eref e )
{
	const char* i = e.generalQ.begin();
	while i != e.generalQ.end() {
		// FuncId* fi = static_cast< FuncId* >( i );
		// i += sizeof( FuncId );
		// i += fi->doOperation( e, i );
		// i += doOperation( *fi, e, i );
		unsigned int op = *static_cast< const unsigned int* >( i );
		i += sizeof( unsigned int );
		i += this->opVec_[ op ]( e, i );
			// opVec is set up statically, has the function ptrs.
			// All are of the form f( Eref e, const char* i ).
	}
}
*/

/*
unsigned int FuncId::doOperation( Eref e, char* i )
{
	unsigned int op = *static_cast< unsigned int* >( i );
	i += sizeof( unsigned int );
	return opVec_[ op ]( i ) + sizeof( unsigned int );
}
*/


void IntFire::setVm( double v )
{
	Vm_ = v;
}

void IntFire::setTau( double v )
{
	tau_ = v;
}

void IntFire::setThresh( double v )
{
	thresh_ = v;
}

