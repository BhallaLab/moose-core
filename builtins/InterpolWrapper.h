#ifndef _InterpolWrapper_h
#define _InterpolWrapper_h
class InterpolWrapper: 
	public Interpol, public Neutral
{
//	friend Element* gateConnInterpolLookup( const Conn* );
	friend Element* lookupInConnInterpolLookup( const Conn* );
    public:
		InterpolWrapper(const string& n)
		:
			Neutral( n ),
			lookupSrc_( &lookupOutConn_ ),
			// gateConn uses a templated lookup function,
			lookupOutConn_( this )
			// lookupInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setXmin( Conn* c, double value ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				localSetXmin( value );
		}
		static double getXmin( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->
				localGetXmin();
		}
		static void setXmax( Conn* c, double value ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				localSetXmax( value );
		}
		static double getXmax( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->
				localGetXmax();
		}
		static void setXdivs( Conn* c, int value ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				localSetXdivs( value );
		}
		static int getXdivs( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->
				localGetXdivs();
		}
		static void setMode( Conn* c, int value ) {
			static_cast< InterpolWrapper* >( c->parent() )->localSetMode( value );
		}
		static int getMode( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->localGetMode();
		}
		// Later do interpolation etc to preseve contents.
		static void setDx( Conn* c, double value ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				localSetDx( value );
		}
		static double getDx( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->
				localGetDx();
		}

		// Scale the table up and down.
		static void setSy( Conn* c, double value ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				localSetSy( value );
		}
		static double getSy( const Element* e ) {
			return static_cast< const InterpolWrapper* >( e )->
				localGetSy();
		}
		static void setTable(
			Element* e, unsigned long index, double value );
		static double getTable(
			const Element* e, unsigned long index );
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getLookupSrc( Element* e ) {
			return &( static_cast< InterpolWrapper* >( e )->lookupSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void lookupFuncLocal( double x ) {
			lookupSrc_.send( doLookup ( x ) );
		}
		static void lookupFunc( Conn* c, double x ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				lookupFuncLocal( x );
		}
		
		static void tabFillFunc( Conn* c, int xdivs, int mode ) {
			static_cast< InterpolWrapper* >( c->parent() )->
				tabFill( xdivs, mode );
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
/*
		static Conn* getGateConn( Element* e ) {
			return &( static_cast< InterpolWrapper* >( e )->gateConn_ );
		}
		*/
		static Conn* getLookupOutConn( Element* e ) {
			return &( static_cast< InterpolWrapper* >( e )->lookupOutConn_ );
		}
		static Conn* getLookupInConn( Element* e ) {
			return &( static_cast< InterpolWrapper* >( e )->lookupInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Interpol* p = dynamic_cast<const Interpol *>(proto);
			// if (p)... and so on. 
			return new InterpolWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > lookupSrc_;
		// UniConn< gateConnInterpolLookup > gateConn_;
		MultiConn lookupOutConn_;
		UniConn< lookupInConnInterpolLookup > lookupInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};

extern unsigned long InterpolOffset();

#endif // _InterpolWrapper_h
