#ifndef _ClockJobWrapper_h
#define _ClockJobWrapper_h

// We maintain a list of these. Each goes out to a number of ClockTicks.
class ClockTickMsgSrc
{
	public:
		ClockTickMsgSrc( Element* e )
			: dt_( 1.0 ), stage_( 0.0 ), nextTime_( 0.0 ),
				nextClockTime_( 0.0 ), conn_( e ), next_( 0 )
		{
			;
		}
		~ClockTickMsgSrc( );

		ClockTickMsgSrc( Element* e, Element* target );
		void setProcInfo( ProcInfo info );
		void start( ProcInfo info, double maxTime );
		double incrementClock( ProcInfo info, double prevClockTime );
		void reinit();
		void resched();
		void connect( ClockTickMsgSrc& dest );
		ClockTickMsgSrc** next() {
			return &next_;
		}

		bool operator<( const ClockTickMsgSrc& other ) const {
			if ( dt_ < other.dt_ ) return 1;
			if ( dt_ == other.dt_ && stage_ < other.stage_ ) return 1;
			return 0;
		}

		bool operator==( const ClockTickMsgSrc& other ) const {
			return ( dt_ == other.dt_ && 
				stage_ == other.stage_ &&
				procFunc_ == other.procFunc_ && 
				reinitFunc_ == other.reinitFunc_ &&
				reschedFunc_ == other.reschedFunc_ );
		}
		double dt() {
			return dt_;
		}

		void updateDt( double newdt, Conn* tick );

		ClockTickMsgSrc* findSrcOf( const Conn* tick );
		// Swaps the order of the ClockTickMsgSrc in their linked list
		void swap( ClockTickMsgSrc** other );

		// Current ClockTickMsgSrc absorbs other. 
		// Require that other->next_ == this.
		void merge( ClockTickMsgSrc* other );

		void updateNextClockTime( );

	private:
		double dt_;
		double stage_;
		double nextTime_;
		double nextClockTime_;
		RecvFunc procFunc_;
		RecvFunc reinitFunc_;
		RecvFunc reschedFunc_;
		PlainMultiConn conn_;
		ClockTickMsgSrc* next_;
		Op1< ProcInfo > op_;
		Element* target_;
};

class ClockJobWrapper: 
	public ClockJob, public JobWrapper
{
	friend Element* startCJInConnLookup( const Conn* );
	friend Element* stepCJInConnLookup( const Conn* );
	friend Element* reinitCJInConnLookup( const Conn* );
	friend Element* reschedCJInConnLookup( const Conn* );
	friend Element* resetCJInConnLookup( const Conn* );
    public:
		ClockJobWrapper(const string& n)
		:
			JobWrapper( n ),
			processSrc_( &clockConn_ ),
			reschedSrc_( &clockConn_ ),
			reinitSrc_( &clockConn_ ),
			finishedSrc_( &finishedOutConn_ ),
			clockConn_( this ),
			finishedOutConn_( this ), 
			tickSrc_( 0 )
			// startInConn uses a templated lookup function,
			// stepInConn uses a templated lookup function,
			// reinitInConn uses a templated lookup function,
			// reschedInConn uses a templated lookup function,
			// resetInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static double getRuntime( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->runTime_;
		}
		// This should also refer to the shortest dt to scale nSteps
		static void setRuntime( Conn* c, double value ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				runTime_ = value;
		}
		static double getCurrentTime( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->currentTime_;
		}
		static int getNSteps( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->nSteps_;
		}
		static void setNSteps( Conn* c, int value ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
			nSteps_ = value;
		}
		static int getCurrentStep( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->currentStep_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessSrc( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->processSrc_ );
		}

		static NMsgSrc* getReschedSrc( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->reschedSrc_ );
		}

		static NMsgSrc* getReinitSrc( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->reinitSrc_ );
		}

		static NMsgSrc* getFinishedSrc( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->finishedSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void startFuncLocal( ProcInfo info );
		static void startFunc( Conn* c, ProcInfo info ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				startFuncLocal( info );
		}

		void stepFuncLocal( ProcInfo info, int nsteps );
		static void stepFunc( Conn* c, ProcInfo info, int nsteps ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				stepFuncLocal( info, nsteps );
		}

		// Used to update the dt and stages of clockticks,
		// without changing their list of targets. Applied when the
		// setclock function is called from the script.
		void dtFuncLocal( double dt, Conn* src );
		static void dtFunc( Conn* c, double dt, Conn* src ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				dtFuncLocal( dt, src );
		}

		void reinitFuncLocal( );
		static void reinitFunc( Conn* c ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void reschedFuncLocal( );
		static void reschedFunc( Conn* c ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				reschedFuncLocal(  );
		}

		void resetFuncLocal(  ) {
			reschedFuncLocal();
			reinitFuncLocal();
		}
		static void resetFunc( Conn* c ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				resetFuncLocal(  );
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getClockConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->clockConn_ );
		}
		static Conn* getFinishedOutConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->finishedOutConn_ );
		}
		static Conn* getStartInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->startInConn_ );
		}
		static Conn* getStepInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->stepInConn_ );
		}
		static Conn* getReinitInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->reinitInConn_ );
		}
		static Conn* getReschedInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->reschedInConn_ );
		}
		static Conn* getResetInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->resetInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const ClockJob* p = dynamic_cast<const ClockJob *>(proto);
			// if (p)... and so on. 
			return new ClockJobWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}

		// Sort the ClockTickMsgSrcs in order of dt and stage.
		void sortTicks( );

    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		NMsgSrc0 reschedSrc_;
		NMsgSrc0 reinitSrc_;
		NMsgSrc0 finishedSrc_;
		// We need to redefine the clockConn_ into a special
		// MultiConn that refers to the tickSrc_ for component conns.
		MultiConn clockConn_;
		MultiConn finishedOutConn_;
		UniConn< startCJInConnLookup > startInConn_;
		UniConn< stepCJInConnLookup > stepInConn_;
		UniConn< reinitCJInConnLookup > reinitInConn_;
		UniConn< reschedCJInConnLookup > reschedInConn_;
		UniConn< resetCJInConnLookup > resetInConn_;


///////////////////////////////////////////////////////
// Other local fields                                //
///////////////////////////////////////////////////////
		ClockTickMsgSrc* tickSrc_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ClockJobWrapper_h
