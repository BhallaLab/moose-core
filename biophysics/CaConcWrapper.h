#ifndef _CaConcWrapper_h
#define _CaConcWrapper_h

class CaConcWrapper: 
	public CaConc, public Neutral
{
	friend Element* processConnCaConcLookup( const Conn* );
	friend Element* currentFractionInConnCaConcLookup( const Conn* );
    public:
		CaConcWrapper(const string& n)
		:
			Neutral( n ),
			concSrc_( &concOutConn_ ),
			// processConn uses a templated lookup function,
			concOutConn_( this ),
			currentInConn_( this ),
			// currentFractionInConn uses a templated lookup function,
			increaseInConn_( this ),
			decreaseInConn_( this ),
			basalInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setCa( Conn* c, double value ) {
			static_cast< CaConcWrapper* >( c->parent() )->Ca_ = value;
		}
		static double getCa( const Element* e ) {
			return static_cast< const CaConcWrapper* >( e )->Ca_;
		}
		static void setCaBasal( Conn* c, double value ) {
			static_cast< CaConcWrapper* >( c->parent() )->CaBasal_ = value;
		}
		static double getCaBasal( const Element* e ) {
			return static_cast< const CaConcWrapper* >( e )->CaBasal_;
		}
		static void setTau( Conn* c, double value ) {
			static_cast< CaConcWrapper* >( c->parent() )->tau_ = value;
		}
		static double getTau( const Element* e ) {
			return static_cast< const CaConcWrapper* >( e )->tau_;
		}
		static void setB( Conn* c, double value ) {
			static_cast< CaConcWrapper* >( c->parent() )->B_ = value;
		}
		static double getB( const Element* e ) {
			return static_cast< const CaConcWrapper* >( e )->B_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getConcSrc( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->concSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void currentFuncLocal( double I ) {
			activation_ += I;
		}
		static void currentFunc( Conn* c, double I ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				currentFuncLocal( I );
		}

		void currentFractionFuncLocal( double I, double fraction ) {
			activation_ += I * fraction;
		}
		static void currentFractionFunc( Conn* c, double I, double fraction ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				currentFractionFuncLocal( I, fraction );
		}

		void increaseFuncLocal( double I ) {
			activation_ += fabs( I );
		}
		static void increaseFunc( Conn* c, double I ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				increaseFuncLocal( I );
		}

		void decreaseFuncLocal( double I ) {
			activation_ -= fabs( I );
		}
		static void decreaseFunc( Conn* c, double I ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				decreaseFuncLocal( I );
		}

		void basalFuncLocal( double value ) {
			CaBasal_ = value;
		}
		static void basalFunc( Conn* c, double value ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				basalFuncLocal( value );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< CaConcWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->processConn_ );
		}
		static Conn* getConcOutConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->concOutConn_ );
		}
		static Conn* getCurrentInConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->currentInConn_ );
		}
		static Conn* getCurrentFractionInConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->currentFractionInConn_ );
		}
		static Conn* getIncreaseInConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->increaseInConn_ );
		}
		static Conn* getDecreaseInConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->decreaseInConn_ );
		}
		static Conn* getBasalInConn( Element* e ) {
			return &( static_cast< CaConcWrapper* >( e )->basalInConn_ );
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
		NMsgSrc1< double > concSrc_;
		UniConn< processConnCaConcLookup > processConn_;
		MultiConn concOutConn_;
		PlainMultiConn currentInConn_;
		UniConn< currentFractionInConnCaConcLookup > currentFractionInConn_;
		PlainMultiConn increaseInConn_;
		PlainMultiConn decreaseInConn_;
		PlainMultiConn basalInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _CaConcWrapper_h
