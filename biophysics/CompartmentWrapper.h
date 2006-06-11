#ifndef _CompartmentWrapper_h
#define _CompartmentWrapper_h
class CompartmentWrapper: 
	public Compartment, public Neutral
{
	friend Element* processConnCompartmentLookup( const Conn* );
	friend Element* initConnCompartmentLookup( const Conn* );
	friend Element* randinjectInConnCompartmentLookup( const Conn* );
    public:
		CompartmentWrapper(const string& n)
		:
			Neutral( n ),
			channelSrc_( &channelConn_ ),
			axialSrc_( &axialConn_ ),
			raxialSrc_( &raxialConn_ ),
			channelReinitSrc_( &channelConn_ ),
			// processConn uses a templated lookup function,
			// initConn uses a templated lookup function,
			channelConn_( this ),
			axialConn_( this ),
			raxialConn_( this ),
			injectInConn_( this )
			// randinjectInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setVm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Vm_ = value;
		}
		static double getVm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Vm_;
		}
		static void setEm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Em_ = value;
		}
		static double getEm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Em_;
		}
		static void setCm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Cm_ = value;
		}
		static double getCm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Cm_;
		}
		static void setRm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Rm_ = value;
		}
		static double getRm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Rm_;
		}
		static void setRa( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Ra_ = value;
		}
		static double getRa( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Ra_;
		}
		static void setIm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Im_ = value;
		}
		static double getIm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Im_;
		}
		static void setInject( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->Inject_ = value;
		}
		static double getInject( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->Inject_;
		}
		static void setInitVm( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->initVm_ = value;
		}
		static double getInitVm( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->initVm_;
		}
		static void setDiameter( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->diameter_ = value;
		}
		static double getDiameter( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->diameter_;
		}
		static void setLength( Conn* c, double value ) {
			static_cast< CompartmentWrapper* >( c->parent() )->length_ = value;
		}
		static double getLength( const Element* e ) {
			return static_cast< const CompartmentWrapper* >( e )->length_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getChannelSrc( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->channelSrc_ );
		}

		static NMsgSrc* getAxialSrc( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->axialSrc_ );
		}

		static NMsgSrc* getRaxialSrc( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->raxialSrc_ );
		}

		static NMsgSrc* getChannelReinitSrc( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->channelReinitSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void channelFuncLocal( double Gk, double Ek ) {
			A_ += Gk * Ek;
			B_ += Gk;
		}
		static void channelFunc( Conn* c, double Gk, double Ek ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				channelFuncLocal( Gk, Ek );
		}

		void raxialFuncLocal( double Ra, double Vm );
		static void raxialFunc( Conn* c, double Ra, double Vm ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				raxialFuncLocal( Ra, Vm );
		}

		void axialFuncLocal( double Vm );
		static void axialFunc( Conn* c, double Vm ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				axialFuncLocal( Vm );
		}

		void injectFuncLocal( double I ) {
			sumInject_ += I;
			Im_ += I;
		}
		static void injectFunc( Conn* c, double I ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				injectFuncLocal( I );
		}

		void randinjectFuncLocal( double prob, double I );
		static void randinjectFunc( Conn* c, double prob, double I ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				randinjectFuncLocal( prob, I );
		}

		void initFuncLocal( ProcInfo info ) {
			axialSrc_.send( Vm_ );
			raxialSrc_.send( Ra_, Vm_ );
		}
		static void initFunc( Conn* c, ProcInfo info ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				initFuncLocal( info );
		}

		void dummyReinitFuncLocal( ProcInfo info ) {
			; 
		}
		static void dummyReinitFunc( Conn* c, ProcInfo info ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				dummyReinitFuncLocal( info );
		}

		void reinitFuncLocal( );
		static void reinitFunc( Conn* c ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				reinitFuncLocal( );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< CompartmentWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->processConn_ );
		}
		static Conn* getInitConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->initConn_ );
		}
		static Conn* getChannelConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->channelConn_ );
		}
		static Conn* getAxialConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->axialConn_ );
		}
		static Conn* getRaxialConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->raxialConn_ );
		}
		static Conn* getInjectInConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->injectInConn_ );
		}
		static Conn* getRandinjectInConn( Element* e ) {
			return &( static_cast< CompartmentWrapper* >( e )->randinjectInConn_ );
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
		NMsgSrc2< double, ProcInfo > channelSrc_;
		NMsgSrc1< double > axialSrc_;
		NMsgSrc2< double, double > raxialSrc_;
		NMsgSrc1< double > channelReinitSrc_;
		UniConn< processConnCompartmentLookup > processConn_;
		UniConn< initConnCompartmentLookup > initConn_;
		MultiConn channelConn_;
		MultiConn axialConn_;
		MultiConn raxialConn_;
		PlainMultiConn injectInConn_;
		UniConn< randinjectInConnCompartmentLookup > randinjectInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _CompartmentWrapper_h
