#ifndef _MoleculeWrapper_h
#define _MoleculeWrapper_h
class MoleculeWrapper: 
	public Molecule, public Neutral
{
	friend Element* processConnMoleculeLookup( const Conn* );
	friend Element* sumProcessInConnMoleculeLookup( const Conn* );
    public:
		MoleculeWrapper(const string& n)
		:
			Neutral( n ),
			reacSrc_( &reacConn_ ),
			nSrc_( &nOutConn_ ),
			// processConn uses a templated lookup function,
			reacConn_( this ),
			nOutConn_( this ),
			prdInConn_( this ),
			sumTotalInConn_( this )
			// sumProcessInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setNInit( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->nInit_ = value;
		}
		static double getNInit( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->nInit_;
		}
		void localSetConcInit( double value ) {
			if ( volumeScale_ > 0 )
				nInit_ = value * volumeScale_;
			else
				nInit_ = value;
		}
		static void setConcInit( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				localSetConcInit( value );
		}
		double localGetConcInit( ) const {
			return nInit_ * volumeScale_;
			if ( volumeScale_ > 0 )
				return nInit_ / volumeScale_;
			else
				return nInit_;
		}
		static double getConcInit( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->
				localGetConcInit();
		}
		static void setVolumeScale( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->volumeScale_ = value;
		}
		static double getVolumeScale( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->volumeScale_;
		}
		static void setN( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->n_ = value;
		}
		static double getN( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->n_;
		}
		void localSetConc( double value ) {
			if ( volumeScale_ > 0 )
				n_ = value * volumeScale_;
			else
				n_ = value;
		}
		static void setConc( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				localSetConc( value );
		}
		double localGetConc( ) const {
			if ( volumeScale_ > 0 )
				return n_ / volumeScale_;
			else
				return n_;
		}
		static double getConc( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->
				localGetConc();
		}
		static void setMode( Conn* c, int value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->mode_ = value;
		}
		static int getMode( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->mode_;
		}
		static void setSlaveEnable( Conn* c, int value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->mode_ =
				value;
		}
		static int getSlaveEnable( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->mode_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getReacSrc( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->reacSrc_ );
		}
		static NMsgSrc* getNSrc( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->nSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void reacFunc( Conn* c, double A, double B ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				reacFuncLocal( A, B );
		}

		static void prdFunc( Conn* c, double A, double B ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				reacFuncLocal( A, B );
		}

		static void sumTotalFunc( Conn* c, double n ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				sumTotalFuncLocal( n );
		}

		void sumProcessFuncLocal( ProcInfo info ) {
			n_ = total_;
			total_ = 0.0;
		}
		static void sumProcessFunc( Conn* c, ProcInfo info ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				sumProcessFuncLocal( info );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->processConn_ );
		}
		static Conn* getReacConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->reacConn_ );
		}
		static Conn* getPrdInConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->prdInConn_ );
		}
		static Conn* getSumTotalInConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->sumTotalInConn_ );
		}
		static Conn* getSumProcessInConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->sumProcessInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Molecule* p = dynamic_cast<const Molecule *>(proto);
			// if (p)... and so on. 
			return new MoleculeWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > reacSrc_;
		NMsgSrc1< double > nSrc_;
		UniConn< processConnMoleculeLookup > processConn_;
		MultiConn reacConn_;
		MultiConn nOutConn_;
		PlainMultiConn prdInConn_;
		PlainMultiConn sumTotalInConn_;
		UniConn< sumProcessInConnMoleculeLookup > sumProcessInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _MoleculeWrapper_h
