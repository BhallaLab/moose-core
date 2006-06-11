#ifndef _SpikeGenWrapper_h
#define _SpikeGenWrapper_h
class SpikeGenWrapper: 
	public SpikeGen, public Neutral
{
	friend Element* channelConnSpikeGenLookup( const Conn* );
    public:
		SpikeGenWrapper(const string& n)
		:
			Neutral( n ),
			eventSrc_( &eventOutConn_ ),
			channelSrc_( &channelConn_ ),
			// channelConn uses a templated lookup function,
			eventOutConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setThreshold( Conn* c, double value ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->threshold_ = value;
		}
		static double getThreshold( const Element* e ) {
			return static_cast< const SpikeGenWrapper* >( e )->threshold_;
		}
		static void setAbsoluteRefractoryPeriod( Conn* c, double value ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->absoluteRefractoryPeriod_ = value;
		}
		static double getAbsoluteRefractoryPeriod( const Element* e ) {
			return static_cast< const SpikeGenWrapper* >( e )->absoluteRefractoryPeriod_;
		}
		static void setAmplitude( Conn* c, double value ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->amplitude_ = value;
		}
		static double getAmplitude( const Element* e ) {
			return static_cast< const SpikeGenWrapper* >( e )->amplitude_;
		}
		static void setState( Conn* c, double value ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->state_ = value;
		}
		static double getState( const Element* e ) {
			return static_cast< const SpikeGenWrapper* >( e )->state_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getEventSrc( Element* e ) {
			return &( static_cast< SpikeGenWrapper* >( e )->eventSrc_ );
		}

		static SingleMsgSrc* getChannelSrc( Element* e ) {
			return &( static_cast< SpikeGenWrapper* >( e )->channelSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void channelFuncLocal( double V, ProcInfo info );
		static void channelFunc( Conn* c, double V, ProcInfo info ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->
				channelFuncLocal( V, info );
		}

		// Set it so that the first spike is allowed.
		void reinitFuncLocal( double Vm ) {
			lastEvent_ = -absoluteRefractoryPeriod_;
		}
		static void reinitFunc( Conn* c, double Vm ) {
			static_cast< SpikeGenWrapper* >( c->parent() )->
				reinitFuncLocal( Vm );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getChannelConn( Element* e ) {
			return &( static_cast< SpikeGenWrapper* >( e )->channelConn_ );
		}
		static Conn* getEventOutConn( Element* e ) {
			return &( static_cast< SpikeGenWrapper* >( e )->eventOutConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const SpikeGen* p = dynamic_cast<const SpikeGen *>(proto);
			// if (p)... and so on. 
			return new SpikeGenWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > eventSrc_;
		SingleMsgSrc2< double, double > channelSrc_;
		UniConn< channelConnSpikeGenLookup > channelConn_;
		MultiConn eventOutConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _SpikeGenWrapper_h
