#ifndef _SchedWrapper_h
#define _SchedWrapper_h
class SchedWrapper: 
	public Sched, public Neutral
{
	friend Element* startInConnLookup( const Conn* );
	friend Element* stopSchedInConnLookup( const Conn* );
    public:
		SchedWrapper(const string& n)
		:
			Neutral( n ),
			processSrc_( &processOutConn_ ),
			processOutConn_( this )
			// startInConn uses a templated lookup function,
			// stopInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessSrc( Element* e ) {
			return &( static_cast< SchedWrapper* >( e )->processSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void startFunc( Conn* c, string job, string shell ) {
			static_cast< SchedWrapper* >( c->parent() )->
				startFuncLocal( job, shell );
		}

		void startFuncLocal( const string& job, const string& shell );

		static void stopFunc( Conn* c, string job ) {
			static_cast< SchedWrapper* >( c->parent() )->
				stopFuncLocal( job );
		}
		void stopFuncLocal( string job ) {
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessOutConn( Element* e ) {
			return &( static_cast< SchedWrapper* >( e )->processOutConn_ );
		}
		static Conn* getStartInConn( Element* e ) {
			return &( static_cast< SchedWrapper* >( e )->startInConn_ );
		}
		static Conn* getStopInConn( Element* e ) {
			return &( static_cast< SchedWrapper* >( e )->stopInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Sched* p = dynamic_cast<const Sched *>(proto);
			// if (p)... and so on. 
			return new SchedWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		MultiConn processOutConn_;
		UniConn< startInConnLookup > startInConn_;
		UniConn< stopSchedInConnLookup > stopInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _SchedWrapper_h
