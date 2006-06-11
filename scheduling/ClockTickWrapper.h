#ifndef _ClockTickWrapper_h
#define _ClockTickWrapper_h
class ClockTickWrapper: 
	public ClockTick, public Neutral
{
	friend Element* clockConnLookup( const Conn* );
    public:
		ClockTickWrapper(const string& n)
		:
			Neutral( n ),
			processSrc_( &tickConn_ ),
			reinitSrc_( &tickConn_ ),
			passStepSrc_( &solverStepConn_ ),
			dtSrc_( &clockConn_ ),
			// clockConn uses a templated lookup function,
			tickConn_( this ),
			solverStepConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		void innerSetDt( double value ) {
			dt_ = value;
			dtSrc_.send( dt_, dtSrc_.conn() );
		}
		static void setDt( Conn* c, double value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				innerSetDt( value );
		}
		static double getDt( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->dt_;
		}
		void innerSetStage( int value ) {
			stage_ = value;
			//dtSrc_.send( dt_, stage_ );
		}
		static void setStage( Conn* c, int value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				innerSetStage( value );
		}
		static int getStage( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->stage_;
		}
		static void setNextt( Conn* c, double value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->nextt_ = value;
		}
		static double getNextt( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->nextt_;
		}
		static void setEpsnextt( Conn* c, double value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->epsnextt_ = value;
		}
		static double getEpsnextt( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->epsnextt_;
		}
		static void setMax_clocks( Conn* c, double value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->max_clocks_ = value;
		}
		static double getMax_clocks( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->max_clocks_;
		}

		void innerSetPath( const string& path );
		static void setPath( Conn* c, string path ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
			innerSetPath( path );
		}

		static string getPath( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->path_;
		}
		static void setNclocks( Conn* c, double value ) {
			static_cast< ClockTickWrapper* >( c->parent() )->nclocks_ = value;
		}
		static double getNclocks( const Element* e ) {
			return static_cast< const ClockTickWrapper* >( e )->nclocks_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessSrc( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->processSrc_ );
		}

		static NMsgSrc* getReinitSrc( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->reinitSrc_ );
		}

		static NMsgSrc* getPassStepSrc( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->passStepSrc_ );
		}
		static SingleMsgSrc* getDtSrc( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->dtSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void checkStepFunc( Conn* c, double t ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				checkStepFuncLocal( t );
		}
		void checkStepFuncLocal( double t ) {
			if( t <= nextt_ )
				passStepSrc_.send( t );
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				processFuncLocal( info );
		}
		void processFuncLocal( ProcInfo info ) {
			processSrc_.send( info );
		}

		void reinitFuncLocal(  ) {
			nextt_ = 0.0;
			epsnextt_ = 0.0;
			reinitSrc_.send();
		}
		static void reinitFunc( Conn* c ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void reschedFuncLocal(  ) {
			innerSetPath( path_ );
		}
		static void reschedFunc( Conn* c ) {
			static_cast< ClockTickWrapper* >( c->parent() )->
				reschedFuncLocal(  );
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getClockConn( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->clockConn_ );
		}
		static Conn* getTickConn( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->tickConn_ );
		}
		static Conn* getSolverStepConn( Element* e ) {
			return &( static_cast< ClockTickWrapper* >( e )->solverStepConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const ClockTick* p = dynamic_cast<const ClockTick *>(proto);
			// if (p)... and so on. 
			return new ClockTickWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		NMsgSrc0 reinitSrc_;
		NMsgSrc1< double > passStepSrc_;
		SingleMsgSrc2< double, Conn* > dtSrc_;
		UniConn< clockConnLookup > clockConn_;
		MultiConn tickConn_;
		MultiConn solverStepConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ClockTickWrapper_h
