#ifndef _SynChanWrapper_h
#define _SynChanWrapper_h
class SynChanWrapper: 
	public SynChan, public Neutral
{
	friend Element* channelConnSynChanLookup( const Conn* );
	friend Element* IkOutConnSynChanLookup( const Conn* );
	friend Element* EkInConnSynChanLookup( const Conn* );
    public:
		SynChanWrapper(const string& n)
		:
			Neutral( n ),
			channelSrc_( &channelConn_ ),
			IkSrc_( &IkOutConn_ ),
			// channelConn uses a templated lookup function,
			// IkOutConn uses a templated lookup function,
			// EkInConn uses a templated lookup function,
			activationInConn_( this ),
			modulatorInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setEk( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->Ek_ = value;
		}
		static double getEk( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->Ek_;
		}
		static void setGk( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->Gk_ = value;
		}
		static double getGk( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->Gk_;
		}
		static void setIk( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->Ik_ = value;
		}
		static double getIk( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->Ik_;
		}
		static void setGbar( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->Gbar_ = value;
		}
		static double getGbar( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->Gbar_;
		}
		static void setTau1( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->tau1_ = value;
		}
		static double getTau1( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->tau1_;
		}
		static void setTau2( Conn* c, double value ) {
			static_cast< SynChanWrapper* >( c->parent() )->tau2_ = value;
		}
		static double getTau2( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->tau2_;
		}
		static void setNormalizeWeights( Conn* c, int value ) {
			static_cast< SynChanWrapper* >( c->parent() )->normalizeWeights_ = value;
		}
		static int getNormalizeWeights( const Element* e ) {
			return static_cast< const SynChanWrapper* >( e )->normalizeWeights_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getChannelSrc( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->channelSrc_ );
		}

		static SingleMsgSrc* getIkSrc( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->IkSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void channelFuncLocal( double Vm, ProcInfo info );
		static void channelFunc( Conn* c, double Vm, ProcInfo info ) {
			static_cast< SynChanWrapper* >( c->parent() )->
				channelFuncLocal( Vm, info );
		}

		void EkFuncLocal( double Ek ) {
			Ek_ = Ek;
		}
		static void EkFunc( Conn* c, double Ek ) {
			static_cast< SynChanWrapper* >( c->parent() )->
				EkFuncLocal( Ek );
		}

		void activationFuncLocal( double act ) {
			activation_ += act;
		}
		static void activationFunc( Conn* c, double act ) {
			static_cast< SynChanWrapper* >( c->parent() )->
				activationFuncLocal( act );
		}

		void modulatorFuncLocal( double mod ) {
			modulation_ *= mod;
		}
		static void modulatorFunc( Conn* c, double mod ) {
			static_cast< SynChanWrapper* >( c->parent() )->
				modulatorFuncLocal( mod );
		}

		void reinitFuncLocal( double Vm );
		static void reinitFunc( Conn* c, double Vm ) {
			static_cast< SynChanWrapper* >( c->parent() )->
				reinitFuncLocal( Vm );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////
		static void setSynapsesValue(
			Element* e , unsigned long index, SynInfo value );
		static SynInfo getSynapsesValue(
			const Element* e , unsigned long index );
		static vector< Conn* >& getSynapsesConn( Element* e ) {
			return reinterpret_cast< vector< Conn* >& >(
				static_cast< SynChanWrapper* >( e )->synapsesConn_
			);
		}
		static unsigned long newSynapsesConn( Element* e );
		static void synapsesFunc( Conn* c, double time );


///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getChannelConn( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->channelConn_ );
		}
		static Conn* getIkOutConn( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->IkOutConn_ );
		}
		static Conn* getEkInConn( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->EkInConn_ );
		}
		static Conn* getActivationInConn( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->activationInConn_ );
		}
		static Conn* getModulatorInConn( Element* e ) {
			return &( static_cast< SynChanWrapper* >( e )->modulatorInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const SynChan* p = dynamic_cast<const SynChan *>(proto);
			// if (p)... and so on. 
			return new SynChanWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc2< double, double > channelSrc_;
		SingleMsgSrc1< double > IkSrc_;
		UniConn< channelConnSynChanLookup > channelConn_;
		UniConn< IkOutConnSynChanLookup > IkOutConn_;
		UniConn< EkInConnSynChanLookup > EkInConn_;
		PlainMultiConn activationInConn_;
		PlainMultiConn modulatorInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////
		vector< SynConn< SynInfo >* > synapsesConn_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _SynChanWrapper_h
