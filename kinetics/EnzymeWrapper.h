#ifndef _EnzymeWrapper_h
#define _EnzymeWrapper_h
class EnzymeWrapper: 
	public Enzyme, public Neutral
{
	friend Element* processConnEnzymeLookup( const Conn* );
	friend Element* enzConnEnzymeLookup( const Conn* );
	friend Element* cplxConnEnzymeLookup( const Conn* );
	friend Element* intramolInConnEnzymeLookup( const Conn* );
    public:
		EnzymeWrapper(const string& n)
		:
			Neutral( n ),
			enzSrc_( &enzConn_ ),
			cplxSrc_( &cplxConn_ ),
			subSrc_( &subConn_ ),
			prdSrc_( &prdOutConn_ ),
			// processConn uses a templated lookup function,
			// enzConn uses a templated lookup function,
			// cplxConn uses a templated lookup function,
			subConn_( this ),
			prdOutConn_( this )
			// intramolInConn uses a templated lookup function
		{
			// We start off in implicit mode so it sort of works right 
			// away. Usually we will set it to explicit mode,
			// as soon as the first reset is done. It can't be done
			// here because it depends on the existence of the parent
			// molecule.
			procFunc_ = &EnzymeWrapper::implicitProcFunc;
			Km_ = ( k2_ + k3_ ) / k1_;
			sA_ = 0;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setK1( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->k1_ = value;
		}
		static double getK1( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->k1_;
		}
		static void setK2( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->k2_ = value;
		}
		static double getK2( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->k2_;
		}
		static void setK3( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->k3_ = value;
		}
		static double getK3( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->k3_;
		}
		static void setKm( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				innerSetKm( value );
		}
		static double getKm( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->Km_;
		}

		void innerSetMode( int mode );
		static void setMode( Conn* c, int value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				innerSetMode( value );
		}

		static int getMode( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->
				innerGetMode();
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getEnzSrc( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->enzSrc_ );
		}

		static SingleMsgSrc* getCplxSrc( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->cplxSrc_ );
		}

		static NMsgSrc* getSubSrc( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->subSrc_ );
		}

		static NMsgSrc* getPrdSrc( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->prdSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  ) {
				/*
			cout << "reinitFuncLocal, first: k3 = " << k3_ << 
				", this = " << this << "\n";;
				*/
			eA_ = pA_ = B_ = e_ = 0.0;
			sA_ = k2_;
			pA_ = k3_;
			s_ = 1.0;
			sk1_ = 1.0;
			/*
			cout << "reinitFuncLocal, second: k3 = " << k3_ << "\n";
			*/
		}
		static void reinitFunc( Conn* c ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
		/*
			cout << "processFunc, first: k3 = " << 
				static_cast< EnzymeWrapper* >( c->parent() )->k3_ <<
				", this = " << c->parent() << "\n";
				*/
			static_cast< EnzymeWrapper* >( c->parent() )->
				processFuncLocal( info );
				/*
			cout << "processFunc, second: k3 = " << 
				static_cast< EnzymeWrapper* >( c->parent() )->k3_ <<
				", this = " << c->parent() << "\n";
				*/
		}

		static void enzFunc( Conn* c, double n ) {
				/*
			cout << "enzFunc, first: k3 = " <<
				static_cast< EnzymeWrapper* >( c->parent())->k3_ <<
				", this = " << c->parent() << "\n";
				*/
			static_cast< EnzymeWrapper* >( c->parent() )->e_ = n;
				/*
			cout << "enzFunc, second: k3 = " <<
				static_cast< EnzymeWrapper* >( c->parent())->k3_ <<
				", this = " << c->parent() << "\n";
				*/
		}

		void cplxFuncLocal( double n ) {
				/*
			cout << "cplxFunc, first: k3 = " << k3_ << 
				", this = " << this << "\n";
				*/
			sA_ *= n;
			pA_ *= n;
		}
		static void cplxFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				cplxFuncLocal( n );
		}

		static void subFunc( Conn* c, double n ) {
				/*
			cout << "subFunc, first: k3 = " <<
				static_cast< EnzymeWrapper* >( c->parent())->k3_ <<
				", this = " << c->parent() << "\n";
				*/
			static_cast< EnzymeWrapper* >( c->parent() )->s_ *= n;
				/*
			cout << "subFunc, second: k3 = " <<
				static_cast< EnzymeWrapper* >( c->parent())->k3_ <<
				", this = " << c->parent() << "\n";
				*/
		}

		void intramolFuncLocal( double n );
		static void intramolFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				intramolFuncLocal( n );
		}

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->processConn_ );
		}
		static Conn* getEnzConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->enzConn_ );
		}
		static Conn* getCplxConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->cplxConn_ );
		}
		static Conn* getSubConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->subConn_ );
		}
		static Conn* getPrdOutConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->prdOutConn_ );
		}
		static Conn* getIntramolInConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->intramolInConn_ );
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
		SingleMsgSrc2< double, double > enzSrc_;
		SingleMsgSrc2< double, double > cplxSrc_;
		NMsgSrc2< double, double > subSrc_;
		NMsgSrc2< double, double > prdSrc_;
		UniConn< processConnEnzymeLookup > processConn_;
		UniConn< enzConnEnzymeLookup > enzConn_;
		UniConn< cplxConnEnzymeLookup > cplxConn_;
		MultiConn subConn_;
		MultiConn prdOutConn_;
		UniConn< intramolInConnEnzymeLookup > intramolInConn_;

///////////////////////////////////////////////////////
// Function definitions                              //
///////////////////////////////////////////////////////
		void ( EnzymeWrapper::*procFunc_ )( );
		void implicitProcFunc();
		void explicitProcFunc();
		void makeComplex();
		int innerGetMode() const {
			return ( procFunc_ == &EnzymeWrapper::implicitProcFunc );
		}

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _EnzymeWrapper_h
