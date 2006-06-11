#ifndef _HHChannelWrapper_h
#define _HHChannelWrapper_h
class HHChannelWrapper: 
	public HHChannel, public Neutral
{
	friend Element* channelConnHHChannelLookup( const Conn* );
	friend Element* xGateConnHHChannelLookup( const Conn* );
	friend Element* yGateConnHHChannelLookup( const Conn* );
	friend Element* zGateConnHHChannelLookup( const Conn* );
	friend Element* IkOutConnHHChannelLookup( const Conn* );
	friend Element* EkInConnHHChannelLookup( const Conn* );
	friend Element* concenInConnHHChannelLookup( const Conn* );
	friend Element* addGbarInConnHHChannelLookup( const Conn* );
	friend Element* reinitInConnHHChannelLookup( const Conn* );
    public:
		HHChannelWrapper(const string& n)
		:
			Neutral( n ),
			channelSrc_( &channelConn_ ),
			IkSrc_( &IkOutConn_ ),
			xGateSrc_( &xGateConn_ ),
			yGateSrc_( &yGateConn_ ),
			zGateSrc_( &zGateConn_ ),
			xGateReinitSrc_( &xGateConn_ ),
			yGateReinitSrc_( &yGateConn_ ),
			zGateReinitSrc_( &zGateConn_ )
			// channelConn uses a templated lookup function,
			// xGateConn uses a templated lookup function,
			// yGateConn uses a templated lookup function,
			// zGateConn uses a templated lookup function,
			// IkOutConn uses a templated lookup function,
			// EkInConn uses a templated lookup function,
			// concenInConn uses a templated lookup function,
			// addGbarInConn uses a templated lookup function,
			// reinitInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setGbar( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Gbar_ = value;
		}
		static double getGbar( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Gbar_;
		}
		static void setEk( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Ek_ = value;
		}
		static double getEk( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Ek_;
		}
		static void setXpower( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Xpower_ = value;
		}
		static double getXpower( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Xpower_;
		}
		static void setYpower( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Ypower_ = value;
		}
		static double getYpower( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Ypower_;
		}
		static void setZpower( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Zpower_ = value;
		}
		static double getZpower( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Zpower_;
		}
		static void setSurface( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->surface_ = value;
		}
		static double getSurface( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->surface_;
		}
		static void setInstant( Conn* c, int value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->instant_ = value;
		}
		static int getInstant( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->instant_;
		}
		static void setGk( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Gk_ = value;
		}
		static double getGk( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Gk_;
		}
		static void setIk( Conn* c, double value ) {
			static_cast< HHChannelWrapper* >( c->parent() )->Ik_ = value;
		}
		static double getIk( const Element* e ) {
			return static_cast< const HHChannelWrapper* >( e )->Ik_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getChannelSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->channelSrc_ );
		}

		static SingleMsgSrc* getIkSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->IkSrc_ );
		}

		static SingleMsgSrc* getXGateSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->xGateSrc_ );
		}

		static SingleMsgSrc* getYGateSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->yGateSrc_ );
		}

		static SingleMsgSrc* getZGateSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->zGateSrc_ );
		}

		static SingleMsgSrc* getXGateReinitSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->xGateReinitSrc_ );
		}

		static SingleMsgSrc* getYGateReinitSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->yGateReinitSrc_ );
		}

		static SingleMsgSrc* getZGateReinitSrc( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->zGateReinitSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void channelFuncLocal( double Vm, ProcInfo info );
		static void channelFunc( Conn* c, double Vm, ProcInfo info ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				channelFuncLocal( Vm, info );
		}

		void EkFuncLocal( double Ek ) {
			Ek_ = Ek;
		}
		static void EkFunc( Conn* c, double Ek ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				EkFuncLocal( Ek );
		}

		void concenFuncLocal( double conc ) {
			conc_ = conc;
		}
		static void concenFunc( Conn* c, double conc ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				concenFuncLocal( conc );
		}

		void addGbarFuncLocal( double gbar ) {
			Gbar_ += gbar;
		}
		static void addGbarFunc( Conn* c, double gbar ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				addGbarFuncLocal( gbar );
		}

		void reinitFuncLocal( double Vm );
		static void reinitFunc( Conn* c, double Vm ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				reinitFuncLocal( Vm );
		}

		void xGateFuncLocal( double X, double gScale ) {
			X_ = X;
			g_ *= gScale;
		}
		static void xGateFunc( Conn* c, double X, double gScale ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				xGateFuncLocal( X, gScale );
		}

		void yGateFuncLocal( double Y, double gScale ) {
			Y_ = Y;
			g_ *= gScale;
		}
		static void yGateFunc( Conn* c, double Y, double gScale ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				yGateFuncLocal( Y, gScale );
		}

		void zGateFuncLocal( double Z, double gScale ) {
			Z_ = Z;
			g_ *= gScale;
		}
		static void zGateFunc( Conn* c, double Z, double gScale ) {
			static_cast< HHChannelWrapper* >( c->parent() )->
				zGateFuncLocal( Z, gScale );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getChannelConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->channelConn_ );
		}
		static Conn* getXGateConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->xGateConn_ );
		}
		static Conn* getYGateConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->yGateConn_ );
		}
		static Conn* getZGateConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->zGateConn_ );
		}
		static Conn* getIkOutConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->IkOutConn_ );
		}
		static Conn* getEkInConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->EkInConn_ );
		}
		static Conn* getConcenInConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->concenInConn_ );
		}
		static Conn* getAddGbarInConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->addGbarInConn_ );
		}
		static Conn* getReinitInConn( Element* e ) {
			return &( static_cast< HHChannelWrapper* >( e )->reinitInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto );

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc2< double, double > channelSrc_;
		SingleMsgSrc1< double > IkSrc_;
		SingleMsgSrc3< double, double, double > xGateSrc_;
		SingleMsgSrc3< double, double, double > yGateSrc_;
		SingleMsgSrc3< double, double, double > zGateSrc_;
		SingleMsgSrc3< double, double, int > xGateReinitSrc_;
		SingleMsgSrc3< double, double, int > yGateReinitSrc_;
		SingleMsgSrc3< double, double, int > zGateReinitSrc_;
		UniConn< channelConnHHChannelLookup > channelConn_;
		UniConn< xGateConnHHChannelLookup > xGateConn_;
		UniConn< yGateConnHHChannelLookup > yGateConn_;
		UniConn< zGateConnHHChannelLookup > zGateConn_;
		UniConn< IkOutConnHHChannelLookup > IkOutConn_;
		UniConn< EkInConnHHChannelLookup > EkInConn_;
		UniConn< concenInConnHHChannelLookup > concenInConn_;
		UniConn< addGbarInConnHHChannelLookup > addGbarInConn_;
		UniConn< reinitInConnHHChannelLookup > reinitInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _HHChannelWrapper_h
