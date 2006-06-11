#ifndef _MyClassWrapper_h
#define _MyClassWrapper_h
class MyClassWrapper: 
	public MyClass, public Neutral
{
	friend Element* distalConnLookup( const Conn* );
	friend Element* processConnLookup( const Conn* );
    public:
		MyClassWrapper(const string& n)
		:
			Neutral( n ),
			axialSrc_( &proximalConn_ ),
			raxialSrc_( &distalConn_ ),
			channelSrc_( &channelOutConn_ ),
			diffusion1Src_( &proximalConn_ ),
			diffusion2Src_( &distalConn_ ),
			proximalConn_( this ),
			// distalConn uses a templated lookup function,
			// processConn uses a templated lookup function,
			channelOutConn_( this ),
			channelInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setVm( Conn* c, double value ) {
			static_cast< MyClassWrapper* >( c->parent() )->Vm_ = value;
		}
		static double getVm( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->Vm_;
		}
		static void setCm( Conn* c, double value ) {
			static_cast< MyClassWrapper* >( c->parent() )->Cm_ = value;
		}
		static double getCm( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->Cm_;
		}
		static void setRm( Conn* c, double value ) {
			static_cast< MyClassWrapper* >( c->parent() )->Rm_ = value;
		}
		static double getRm( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->Rm_;
		}
		static double getPi( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->pi_;
		}
		static double getRa( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->Ra_;
		}
		static void setInject( Conn* c, double value ) {
			static_cast< MyClassWrapper* >( c->parent() )->inject_ = value;
		}
		static double getInject( const Element* e ) {
			return static_cast< const MyClassWrapper* >( e )->inject_;
		}
		static void setCoords(
			Element* e, unsigned long index, double value );
		static double getCoords(
			const Element* e, unsigned long index );
		static void setValues(
			Element* e, unsigned long index, double value );
		static double getValues(
			const Element* e, unsigned long index );
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getAxialSrc( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->axialSrc_ );
		}

		static SingleMsgSrc* getRaxialSrc( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->raxialSrc_ );
		}

		static NMsgSrc* getChannelSrc( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->channelSrc_ );
		}

		static NMsgSrc* getDiffusion1Src( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->diffusion1Src_ );
		}

		static SingleMsgSrc* getDiffusion2Src( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->diffusion2Src_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void axialFunc( Conn* c, double V ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				axialFuncLocal( V );
		}
		void axialFuncLocal( double V ) {
			I_ += (V - Vm_) / Ra_;
		}
		static void raxialFunc( Conn* c, double V, double Ra ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				raxialFuncLocal( V, Ra );
		}
		void raxialFuncLocal( double V, double Ra ) {
			I_ += ( V - Vm_ ) / Ra;
		}
		static void channelFunc( Conn* c, double Gk, double Ek ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				channelFuncLocal( Gk, Ek );
		}
		void channelFuncLocal( double Gk, double Ek ) {
			I_ += (Vm_ - Ek) * Gk;
			Ca_ += I_ * volscale_;
		}
		static void diffusion2Func( Conn* c, double conc, double flux ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				diffusion2FuncLocal( conc, flux );
		}
		void diffusion2FuncLocal( double conc, double flux ) {
			Ca_ += flux;
		}
		static void diffusion1Func( Conn* c, double conc, double flux ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				diffusion1FuncLocal( conc, flux );
		}
		void diffusion1FuncLocal( double conc, double flux ) {
			Ca_ += flux;
		}
		static void processFunc( Conn* c, ProcArg a ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				processFuncLocal( a );
		}
		void processFuncLocal( ProcArg a ) {
			channelSrc_.send( Vm_, a );
			axialSrc_.send( Vm_ );
			raxialSrc_.send( Vm_, Rm_ );
			Vm_ += 0; 
		}
		static void resetFunc( Conn* c ) {
			static_cast< MyClassWrapper* >( c->parent() )->
				resetFuncLocal(  );
		}
		void resetFuncLocal(  ) {
			Vm_ = Erest_;
			I_ = 0;
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////
		static void setInhibValue(
			Element* e , unsigned long index, int value );
		static int getInhibValue(
			const Element* e , unsigned long index );
		static vector< Conn* >& getInhibConn( Element* e ) {
			return reinterpret_cast< vector< Conn* >& >(
				static_cast< MyClassWrapper* >( e )->inhibConn_
			);
		}
		static unsigned long newInhibConn( Element* e );
		static void inhibFunc( Conn* c, double delay );

		static void setExciteValue(
			Element* e , unsigned long index, SynInfo value );
		static SynInfo getExciteValue(
			const Element* e , unsigned long index );
		static vector< Conn* >& getExciteConn( Element* e ) {
			return reinterpret_cast< vector< Conn* >& >(
				static_cast< MyClassWrapper* >( e )->exciteConn_
			);
		}
		static unsigned long newExciteConn( Element* e );
		static void exciteFunc( Conn* c, double delay );


///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProximalConn( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->proximalConn_ );
		}
		static Conn* getDistalConn( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->distalConn_ );
		}
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->processConn_ );
		}
		static Conn* getChannelOutConn( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->channelOutConn_ );
		}
		static Conn* getChannelInConn( Element* e ) {
			return &( static_cast< MyClassWrapper* >( e )->channelInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const MyClass* p = dynamic_cast<const MyClass *>(proto);
			// if (p)... and so on. 
			return new MyClassWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > axialSrc_;
		SingleMsgSrc2< double, double > raxialSrc_;
		NMsgSrc2< double, ProcArg > channelSrc_;
		NMsgSrc2< double, double > diffusion1Src_;
		SingleMsgSrc2< double, double > diffusion2Src_;
		MultiConn proximalConn_;
		UniConn< distalConnLookup > distalConn_;
		UniConn< processConnLookup > processConn_;
		MultiConn channelOutConn_;
		PlainMultiConn channelInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////
		vector< SynConn< int >* > inhibConn_;
		vector< SynConn< SynInfo >* > exciteConn_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _MyClassWrapper_h
