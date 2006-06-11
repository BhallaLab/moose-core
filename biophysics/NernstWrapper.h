#ifndef _NernstWrapper_h
#define _NernstWrapper_h
class NernstWrapper: 
	public Nernst, public Neutral
{
	friend Element* CinInConnNernstLookup( const Conn* );
	friend Element* CoutInConnNernstLookup( const Conn* );
    public:
		NernstWrapper(const string& n)
		:
			Neutral( n ),
			ESrc_( &EOutConn_ ),
			EOutConn_( this )
			// CinInConn uses a templated lookup function,
			// CoutInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		// Temporary while I sort out the issue of ReadOnlyValues
		static void setE( Conn* c, double value ) {
			static_cast< NernstWrapper* >( c->parent() )->E_ = value;
		}
		static double getE( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->E_;
		}
		static void setTemperature( Conn* c, double value ) {
			static_cast< NernstWrapper* >( c->parent() )->
				localSetTemperature( value );
		}
		static double getTemperature( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->Temperature_;
		}
		static void setValence( Conn* c, int value ) {
			static_cast< NernstWrapper* >( c->parent() )->
				localSetValence( value );
		}
		static int getValence( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->valence_;
		}
		static void setCin( Conn* c, double value ) {
			static_cast< NernstWrapper* >( c->parent() )->Cin_ = value;
		}
		static double getCin( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->Cin_;
		}
		static void setCout( Conn* c, double value ) {
			static_cast< NernstWrapper* >( c->parent() )->Cout_ = value;
		}
		static double getCout( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->Cout_;
		}
		static void setScale( Conn* c, double value ) {
			static_cast< NernstWrapper* >( c->parent() )->scale_ = value;
		}
		static double getScale( const Element* e ) {
			return static_cast< const NernstWrapper* >( e )->scale_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getESrc( Element* e ) {
			return &( static_cast< NernstWrapper* >( e )->ESrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void CinFuncLocal( double conc );
		static void CinFunc( Conn* c, double conc ) {
			static_cast< NernstWrapper* >( c->parent() )->
				CinFuncLocal( conc );
		}

		void CoutFuncLocal( double conc );
		static void CoutFunc( Conn* c, double conc ) {
			static_cast< NernstWrapper* >( c->parent() )->
				CoutFuncLocal( conc );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getEOutConn( Element* e ) {
			return &( static_cast< NernstWrapper* >( e )->EOutConn_ );
		}
		static Conn* getCinInConn( Element* e ) {
			return &( static_cast< NernstWrapper* >( e )->CinInConn_ );
		}
		static Conn* getCoutInConn( Element* e ) {
			return &( static_cast< NernstWrapper* >( e )->CoutInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Nernst* p = dynamic_cast<const Nernst *>(proto);
			// if (p)... and so on. 
			return new NernstWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > ESrc_;
		MultiConn EOutConn_;
		UniConn< CinInConnNernstLookup > CinInConn_;
		UniConn< CoutInConnNernstLookup > CoutInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _NernstWrapper_h
