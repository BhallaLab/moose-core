/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#define mtrandf(l,h)     (l)==(h) ? (l) : mtrand() * ((h)-(l)) + (l)

#ifndef _RandomSpikeWrapper_h
#define _RandomSpikeWrapper_h
class RandomSpikeWrapper: 
	public RandomSpike, public Neutral
{
	friend Element* processConnRandomSpikeLookup( const Conn* );
	friend Element* rateInConnRandomSpikeLookup( const Conn* );
    public:
		RandomSpikeWrapper(const string& n)
		:
			Neutral( n ),
			stateSrc_( &stateOutConn_ ),
			stateTimeSrc_( &stateTimeOutConn_ ),
			// processConn uses a templated lookup function,
			stateOutConn_( this ),
			stateTimeOutConn_( this )
			// rateInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setRate( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->rate_ = value;
		}
		static double getRate( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->rate_;
		}
		static void setAbsoluteRefractoryPeriod( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->absoluteRefractoryPeriod_ = value;
		}
		static double getAbsoluteRefractoryPeriod( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->absoluteRefractoryPeriod_;
		}
		static void setState( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->state_ = value;
		}
		static double getState( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->state_;
		}
		static void setReset( Conn* c, int value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->reset_ = value;
		}
		static int getReset( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->reset_;
		}
		static void setResetValue( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->resetValue_ = value;
		}
		static double getResetValue( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->resetValue_;
		}
		static void setMinAmp( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->minAmp_ = value;
		}
		static double getMinAmp( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->minAmp_;
		}
		static void setMaxAmp( Conn* c, double value ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->maxAmp_ = value;
		}
		static double getMaxAmp( const Element* e ) {
			return static_cast< const RandomSpikeWrapper* >( e )->maxAmp_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getStateSrc( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->stateSrc_ );
		}

		static NMsgSrc* getStateTimeSrc( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->stateTimeSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void rateFuncLocal( double rate );
		static void rateFunc( Conn* c, double rate ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->
				rateFuncLocal( rate );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< RandomSpikeWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->processConn_ );
		}
		static Conn* getStateOutConn( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->stateOutConn_ );
		}
		static Conn* getStateTimeOutConn( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->stateTimeOutConn_ );
		}
		static Conn* getRateInConn( Element* e ) {
			return &( static_cast< RandomSpikeWrapper* >( e )->rateInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const RandomSpike* p = dynamic_cast<const RandomSpike *>(proto);
			// if (p)... and so on. 
			return new RandomSpikeWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > stateSrc_;
		NMsgSrc2< double, double > stateTimeSrc_;
		UniConn< processConnRandomSpikeLookup > processConn_;
		MultiConn stateOutConn_;
		MultiConn stateTimeOutConn_;
		UniConn< rateInConnRandomSpikeLookup > rateInConn_;

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
#endif // _RandomSpikeWrapper_h
