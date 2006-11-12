/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DummyDelayWrapper_h
#define _DummyDelayWrapper_h
class DummyDelayWrapper: 
	public DummyDelay, public Neutral
{
	friend Element* processConnDummyDelayLookup( const Conn* );
    public:
		DummyDelayWrapper(const string& n)
		:
			Neutral( n ),
			spikeSrc_( &spikeOutConn_ ),
			spikeTimeSrc_( &spikeTimeOutConn_ ),
			// processConn uses a templated lookup function,
			spikeOutConn_( this ),
			spikeTimeOutConn_( this ),
			spikeInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setThreshold( Conn* c, double value ) {
			static_cast< DummyDelayWrapper* >( c->parent() )->threshold_ = value;
		}
		static double getThreshold( const Element* e ) {
			return static_cast< const DummyDelayWrapper* >( e )->threshold_;
		}
		static void setDelay( Conn* c, int value ) {
			static_cast< DummyDelayWrapper* >( c->parent() )->delay_ = value;
		}
		static int getDelay( const Element* e ) {
			return static_cast< const DummyDelayWrapper* >( e )->delay_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getSpikeSrc( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->spikeSrc_ );
		}

		static NMsgSrc* getSpikeTimeSrc( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->spikeTimeSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void spikeFuncLocal( double amplitude );
		static void spikeFunc( Conn* c, double amplitude ) {
			static_cast< DummyDelayWrapper* >( c->parent() )->
				spikeFuncLocal( amplitude );
		}

		void reinitFuncLocal(  ) {
			amplitude_ = 0;
			stepsRemaining_ = -1;
		}
		static void reinitFunc( Conn* c ) {
			static_cast< DummyDelayWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< DummyDelayWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->processConn_ );
		}
		static Conn* getSpikeOutConn( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->spikeOutConn_ );
		}
		static Conn* getSpikeTimeOutConn( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->spikeTimeOutConn_ );
		}
		static Conn* getSpikeInConn( Element* e ) {
			return &( static_cast< DummyDelayWrapper* >( e )->spikeInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const DummyDelay* p = dynamic_cast<const DummyDelay *>(proto);
			// if (p)... and so on. 
			return new DummyDelayWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > spikeSrc_;
		NMsgSrc2< double, double > spikeTimeSrc_;
		UniConn< processConnDummyDelayLookup > processConn_;
		MultiConn spikeOutConn_;
		MultiConn spikeTimeOutConn_;
		PlainMultiConn spikeInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _DummyDelayWrapper_h
