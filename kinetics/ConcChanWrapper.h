#ifndef _ConcChanWrapper_h
#define _ConcChanWrapper_h
class ConcChanWrapper: 
	public ConcChan, public Neutral
{
	friend Element* processConnConcChanLookup( const Conn* );
	friend Element* influxConnConcChanLookup( const Conn* );
	friend Element* effluxConnConcChanLookup( const Conn* );
	friend Element* nInConnConcChanLookup( const Conn* );
	friend Element* VmInConnConcChanLookup( const Conn* );
    public:
		ConcChanWrapper(const string& n)
		:
			Neutral( n ),
			influxSrc_( &influxConn_ ),
			effluxSrc_( &effluxConn_ )
			// processConn uses a templated lookup function,
			// influxConn uses a templated lookup function,
			// effluxConn uses a templated lookup function,
			// nInConn uses a templated lookup function,
			// VmInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setPermeability( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->permeability_ = value;
		}
		static double getPermeability( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->permeability_;
		}
		static void setN( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->n_ = value;
		}
		static double getN( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->n_;
		}
		static void setVm( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->Vm_ = value;
		}
		static double getVm( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->Vm_;
		}
		static void setENernst( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->ENernst_ = value;
		}
		static double getENernst( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->ENernst_;
		}
		static void setValence( Conn* c, int value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->valence_ = value;
		}
		static int getValence( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->valence_;
		}
		static void setTemperature( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->temperature_ = value;
		}
		static double getTemperature( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->temperature_;
		}
		static void setInVol( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->inVol_ = value;
		}
		static double getInVol( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->inVol_;
		}
		static void setOutVol( Conn* c, double value ) {
			static_cast< ConcChanWrapper* >( c->parent() )->outVol_ = value;
		}
		static double getOutVol( const Element* e ) {
			return static_cast< const ConcChanWrapper* >( e )->outVol_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getInfluxSrc( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->influxSrc_ );
		}

		static SingleMsgSrc* getEffluxSrc( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->effluxSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void influxFuncLocal( double n ) {
			A_ = n;
		}
		static void influxFunc( Conn* c, double n ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				influxFuncLocal( n );
		}

		void effluxFuncLocal( double n ) {
			B_ = n;
		}
		static void effluxFunc( Conn* c, double n ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				effluxFuncLocal( n );
		}

		void nFuncLocal( double n ) {
			n_ = n;
		}
		static void nFunc( Conn* c, double n ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				nFuncLocal( n );
		}

		void VmFuncLocal( double Vm ) {
			Vm_ = Vm;
		}
		static void VmFunc( Conn* c, double Vm ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				VmFuncLocal( Vm );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< ConcChanWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->processConn_ );
		}
		static Conn* getInfluxConn( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->influxConn_ );
		}
		static Conn* getEffluxConn( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->effluxConn_ );
		}
		static Conn* getNInConn( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->nInConn_ );
		}
		static Conn* getVmInConn( Element* e ) {
			return &( static_cast< ConcChanWrapper* >( e )->VmInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const ConcChan* p = dynamic_cast<const ConcChan *>(proto);
			// if (p)... and so on. 
			return new ConcChanWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc2< double, double > influxSrc_;
		SingleMsgSrc2< double, double > effluxSrc_;
		UniConn< processConnConcChanLookup > processConn_;
		UniConn< influxConnConcChanLookup > influxConn_;
		UniConn< effluxConnConcChanLookup > effluxConn_;
		UniConn< nInConnConcChanLookup > nInConn_;
		UniConn< VmInConnConcChanLookup > VmInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ConcChanWrapper_h
