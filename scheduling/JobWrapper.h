#ifndef _JobWrapper_h
#define _JobWrapper_h
class JobWrapper: 
	public Job, public Neutral
{
	friend Element* processInConnLookup( const Conn* );
	friend Element* stopJobInConnLookup( const Conn* );
	friend Element* sleepInConnLookup( const Conn* );
	friend Element* wakeInConnLookup( const Conn* );
    public:
		JobWrapper(const string& n)
		:
			Neutral( n ),
			processSrc_( &processOutConn_ ),
			triggerSrc_( &triggerOutConn_ ),
			processOutConn_( this ),
			triggerOutConn_( this )
			// processInConn uses a templated lookup function,
			// stopInConn uses a templated lookup function,
			// sleepInConn uses a templated lookup function,
			// wakeInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static int getRunning( const Element* e ) {
			return static_cast< const JobWrapper* >( e )->running_;
		}
		static void setDoLoop( Conn* c, int value ) {
			static_cast< JobWrapper* >( c->parent() )->doLoop_ = value;
		}
		static int getDoLoop( const Element* e ) {
			return static_cast< const JobWrapper* >( e )->doLoop_;
		}
		static void setDoTiming( Conn* c, int value ) {
			static_cast< JobWrapper* >( c->parent() )->doTiming_ = value;
		}
		static int getDoTiming( const Element* e ) {
			return static_cast< const JobWrapper* >( e )->doTiming_;
		}
		static void setRealTimeInterval( Conn* c, double value ) {
			static_cast< JobWrapper* >( c->parent() )->realTimeInterval_ = value;
		}
		static double getRealTimeInterval( const Element* e ) {
			return static_cast< const JobWrapper* >( e )->realTimeInterval_;
		}
		static void setPriority( Conn* c, int value ) {
			static_cast< JobWrapper* >( c->parent() )->priority_ = value;
		}
		static int getPriority( const Element* e ) {
			return static_cast< const JobWrapper* >( e )->priority_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessSrc( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->processSrc_ );
		}

		static NMsgSrc* getTriggerSrc( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->triggerSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< JobWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		// Overridden by the ClockJob
		virtual void processFuncLocal( ProcInfo info );

		static void stopFunc( Conn* c ) {
			static_cast< JobWrapper* >( c->parent() )->
				stopFuncLocal(  );
		}
		void stopFuncLocal(  ) {
			terminate_ = 1;
		}
		static void sleepFunc( Conn* c, double time ) {
			static_cast< JobWrapper* >( c->parent() )->
				sleepFuncLocal( time );
		}
		void sleepFuncLocal( double time ) {
			// wakeUpTime_ = time + currentRealTime();
		}
		static void wakeFunc( Conn* c ) {
			static_cast< JobWrapper* >( c->parent() )->
				wakeFuncLocal(  );
		}
		void wakeFuncLocal(  ) {
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessOutConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->processOutConn_ );
		}
		static Conn* getTriggerOutConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->triggerOutConn_ );
		}
		static Conn* getProcessInConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->processInConn_ );
		}
		static Conn* getStopInConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->stopInConn_ );
		}
		static Conn* getSleepInConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->sleepInConn_ );
		}
		static Conn* getWakeInConn( Element* e ) {
			return &( static_cast< JobWrapper* >( e )->wakeInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Job* p = dynamic_cast<const Job *>(proto);
			// if (p)... and so on. 
			return new JobWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		NMsgSrc0 triggerSrc_;
		MultiConn processOutConn_;
		MultiConn triggerOutConn_;
		UniConn< processInConnLookup > processInConn_;
		UniConn< stopJobInConnLookup > stopInConn_;
		UniConn< sleepInConnLookup > sleepInConn_;
		UniConn< wakeInConnLookup > wakeInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _JobWrapper_h
