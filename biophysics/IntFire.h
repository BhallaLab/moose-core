#ifndef _INT_FIRE_H
#define _INT_FIRE_H

/*
class SynInfo
{
	public:
		SynInfo() 
			: weight( 1.0 ), delay( 0.0 )
		{
			;
		}

		SynInfo( double w, double d ) 
			: weight( w ), delay( d )
		{
			;
		}

		SynInfo( const SynInfo& other, double time )
			: weight( other.weight ), delay( time + other.delay )
		{
			;
		}

		// This is backward because the syntax of the priority
		// queue puts the _largest_ element on top.
		bool operator< ( const SynInfo& other ) const {
			return delay > other.delay;
		}
		
		bool operator== ( const SynInfo& other ) const {
			return delay == other.delay && weight == other.weight;
		}

		SynInfo event( double time ) {
			return SynInfo( weight, time + delay );
		}

		double weight;
		double delay;
};
*/

class IntFire: public Data
{
	friend void testStandaloneIntFire();
	friend void testSynapse();
	public: 
		IntFire();
		IntFire( double thresh, double tau );
		void process( const ProcInfo* p, const Eref& e );
		void reinit( Eref& e );

		/**
 		 * Inserts an event into the pendingEvents queue for spikes.
 		 */
		void addSpike( Eref& e, const Qinfo* q, const double& time );
		
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		void setVm( const double& v );
		const double &getVm() const;
		void setTau( const double& v );
		const double &getTau() const;
		void setThresh( const double& v );
		const double &getThresh() const;
		static const Cinfo* initCinfo();
	private:
		double Vm_; // State variable: Membrane potential. Resting pot is 0.
		double thresh_;	// Firing threshold
		double tau_; // Time course of membrane settling.
		vector< SynInfo > synapses_;
		priority_queue< SynInfo > pendingEvents_;
};

#endif // _INT_FIRE_H
