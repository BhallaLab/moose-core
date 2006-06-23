#ifndef _HHGateWrapper_h
#define _HHGateWrapper_h

class HHGateWrapper: public HHGate, public Neutral
{
    public:
		HHGateWrapper(const string& n)
		:
			Neutral( n ),
			gateConn_( this )
		{
			;
		}
		void test();
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setPower( Conn* c, double value ) {
			static_cast< HHGateWrapper* >( c->parent() )->power_ = value;
		}
		static double getPower( const Element* e ) {
			return static_cast< const HHGateWrapper* >( e )->power_;
		}
		static void setState( Conn* c, double value ) {
			static_cast< HHGateWrapper* >( c->parent() )->state_ = value;
		}
		static double getState( const Element* e ) {
			return static_cast< const HHGateWrapper* >( e )->state_;
		}

		static void setInstant( Conn* c, int value ) {
			static_cast< HHGateWrapper* >( c->parent() )->instant_ = value;
		}
		static int getInstant( const Element* e ) {
			return static_cast< const HHGateWrapper* >( e )->instant_;
		}

		static void setA( Conn* c, Interpol value ) {
			static_cast< HHGateWrapper* >( c->parent() )->A_ = value;
		}
		static Interpol getA( const Element* e ) {
			return static_cast< const HHGateWrapper* >( e )->A_;
		}
		/*
		static void setA( Conn* c, Interpol* value ) {
			static_cast< HHGateWrapper* >( c->parent() )->A_ = *value;
		}
		static const Interpol* getA( const Element* e ) {
			return &( static_cast< const HHGateWrapper* >( e )->A_ );
		}
		*/
		static Element* lookupA( Element* e, unsigned long index )
		{
			static InterpolWrapper iw("temp");
			static Interpol* ip = &iw;
			static const unsigned long OFFSET = 
				( unsigned long )( ip ) - ( unsigned long )(&iw);
				// FIELD_OFFSET( InterpolWrapper, Interpol );
			// cerr << "in lookupA (the interpol ): OFFSET = " << OFFSET << "\n";
			return reinterpret_cast< InterpolWrapper* >( 
				( unsigned long )
				( &static_cast< const HHGateWrapper* >( e )->A_ ) -
				OFFSET
				);
		}

		static void setB( Conn* c, Interpol value ) {
			static_cast< HHGateWrapper* >( c->parent() )->B_ = value;
		}
		static Interpol getB( const Element* e ) {
			return static_cast< const HHGateWrapper* >( e )->B_;
		}
		static Element* lookupB( Element* e, unsigned long index )
		{
			static InterpolWrapper iw("temp");
			static Interpol* ip = &iw;
			static const unsigned long OFFSET = 
				( unsigned long )( ip ) - ( unsigned long )(&iw);
			return reinterpret_cast< InterpolWrapper* >( 
				( unsigned long )
				( &static_cast< const HHGateWrapper* >( e )->B_ ) -
				OFFSET
				);
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void gateFuncLocal( Conn* c, double v, double state, double dt);

		static void gateFunc( Conn* c, double v, double state, double dt ) {
			static_cast< HHGateWrapper* >( c->parent() )->
				gateFuncLocal( c, v, state, dt );
		}

		void reinitFuncLocal( Conn* c, double Vm, double power,
			int instant );
		static void reinitFunc( Conn* c, double Vm, double power,
			int instant ) {
			static_cast< HHGateWrapper* >( c->parent() )->
				reinitFuncLocal( c, Vm, power, instant );
		}

		void tabFillFuncLocal( Conn* c, int xdivs, int mode ) {
			A_.tabFill( xdivs, mode );
			B_.tabFill( xdivs, mode );
		}
		static void tabFillFunc( Conn* c, int xdivs, int mode ) {
			static_cast< HHGateWrapper* >( c->parent() )->
				tabFillFuncLocal( c, xdivs, mode );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getGateConn( Element* e ) {
			return &( static_cast< HHGateWrapper* >( e )->gateConn_ );
		}
		static MultiReturnConn* getGateMultiReturnConn( Element* e ) {
			return &( static_cast< HHGateWrapper* >( e )->gateConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const HHGate* p = dynamic_cast<const HHGate *>(proto);
			// if (p)... and so on. 
			return new HHGateWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		MultiReturnConn gateConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _HHGateWrapper_h
