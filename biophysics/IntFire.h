#ifndef _SynInfo_h
#define _SynInfo_h

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

#endif // _SynInfo_h

class IntFire: public Data
{
	friend void testStandaloneIntFire();
	friend void testSynapse();
	public: 
		IntFire( double thresh, double tau );
		void process( const ProcInfo* p, Eref e );
		void reinit( Eref e );

		/**
 		 * Inserts an event into the pendingEvents queue for spikes.
 		 */
		void addSpike( unsigned int id, double time );
		//
		////////////////////////////////////////////////////////////////
		// Field assignment stuff.
		////////////////////////////////////////////////////////////////
		
		void setVm( double v );
		void setTau( double v );
		void setThresh( double v );
		Finfo** initClassInfo();
	private:
		double Vm_; // State variable: Membrane potential. Resting pot is 0.
		double thresh_;	// Firing threshold
		double tau_; // Time course of membrane settling.
		vector< SynInfo > synapses_;
		priority_queue< SynInfo > pendingEvents_;
};
