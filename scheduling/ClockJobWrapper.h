/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _ClockJobWrapper_h
#define _ClockJobWrapper_h
class ClockJobWrapper: 
	public ClockJob, public JobWrapper
{
	friend Element* startInConnClockJobLookup( const Conn* );
	friend Element* stepInConnClockJobLookup( const Conn* );
	friend Element* reinitInConnClockJobLookup( const Conn* );
	friend Element* reschedInConnClockJobLookup( const Conn* );
	friend Element* resetInConnClockJobLookup( const Conn* );
	friend Element* schedNewObjectInConnClockJobLookup( const Conn* );
    public:
		ClockJobWrapper(const string& n)
		:
			JobWrapper( n ),
			processSrc_( &clockConn_ ),
			reschedSrc_( &clockConn_ ),
			reinitSrc_( &clockConn_ ),
			schedNewObjectSrc_( &clockConn_ ),
			finishedSrc_( &finishedOutConn_ ),
			clockConn_( this ),
			finishedOutConn_( this ),
			tickSrc_( 0 )
			// startInConn uses a templated lookup function,
			// stepInConn uses a templated lookup function,
			// reinitInConn uses a templated lookup function,
			// reschedInConn uses a templated lookup function,
			// resetInConn uses a templated lookup function,
			// schedNewObjectInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setRunTime( Conn* c, double value ) {
			static_cast< ClockJobWrapper* >( c->parent() )->runTime_ = value;
		}
		static double getRunTime( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->runTime_;
		}
		static double getCurrentTime( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->currentTime_;
		}
		static void setNSteps( Conn* c, int value ) {
			static_cast< ClockJobWrapper* >( c->parent() )->nSteps_ = value;
		}
		static int getNSteps( const Element* e ) {
			return static_cast< const ClockJobWrapper* >( e )->nSteps_;
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

		static NMsgSrc* getSchedNewObjectSrc( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->schedNewObjectSrc_ );
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

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void reschedFuncLocal(  );
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

		void dtFuncLocal( double dt, Conn* tick );
		static void dtFunc( Conn* c, double dt, Conn* tick ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				dtFuncLocal( dt, tick );
		}

		void schedNewObjectFuncLocal( Element* object ) {
			schedNewObjectSrc_.send( object );
		}
		static void schedNewObjectFunc( Conn* c, Element* object ) {
			static_cast< ClockJobWrapper* >( c->parent() )->
				schedNewObjectFuncLocal( object );
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
		static Conn* getSchedNewObjectInConn( Element* e ) {
			return &( static_cast< ClockJobWrapper* >( e )->schedNewObjectInConn_ );
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


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		NMsgSrc0 reschedSrc_;
		NMsgSrc0 reinitSrc_;
		NMsgSrc1< Element* > schedNewObjectSrc_;
		NMsgSrc0 finishedSrc_;
		MultiConn clockConn_;
		MultiConn finishedOutConn_;
		UniConn< startInConnClockJobLookup > startInConn_;
		UniConn< stepInConnClockJobLookup > stepInConn_;
		UniConn< reinitInConnClockJobLookup > reinitInConn_;
		UniConn< reschedInConnClockJobLookup > reschedInConn_;
		UniConn< resetInConnClockJobLookup > resetInConn_;
		UniConn< schedNewObjectInConnClockJobLookup > schedNewObjectInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		ClockTickMsgSrc* tickSrc_;
		void sortTicks();

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ClockJobWrapper_h
