/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _KsolveWrapper_h
#define _KsolveWrapper_h
class KsolveWrapper: 
	public Ksolve, public Neutral
{
	friend Element* processInConnKsolveLookup( const Conn* );
	friend Element* reinitInConnKsolveLookup( const Conn* );
    public:
		KsolveWrapper(const string& n)
		:
			Neutral( n ),
			processSrc_( &molSolveConn_ ),
			reinitSrc_( &molSolveConn_ ),
			molSrc_( &molSolveConn_ ),
			bufSrc_( &molSolveConn_ ),
			sumTotSrc_( &molSolveConn_ ),
			processReacSrc_( &reacSolveConn_ ),
			reinitReacSrc_( &reacSolveConn_ ),
			processEnzSrc_( &enzSolveConn_ ),
			reinitEnzSrc_( &enzSolveConn_ ),
			processMmEnzSrc_( &mmEnzSolveConn_ ),
			reinitMmEnzSrc_( &mmEnzSolveConn_ ),
			processRateSrc_( &rateSolveConn_ ),
			reinitRateSrc_( &rateSolveConn_ ),
			rateSrc_( &rateSolveConn_ ),
			molSolveConn_( this ),
			// bufSolveConn_( this ),
			// sumTotSolveConn_( this ),
			reacSolveConn_( this ),
			enzSolveConn_( this ),
			mmEnzSolveConn_( this ),
			tabSolveConn_( this ),
			rateSolveConn_( this )
			// processInConn uses a templated lookup function,
			// reinitInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		string localGetPath() const;
		static string getPath( const Element* e ) {
			return static_cast< const KsolveWrapper* >( e )->
			localGetPath();
		}
		void localSetPath( const string& value );
		static void setPath( Conn* c, string value ) {
			static_cast< KsolveWrapper* >( c->parent() )->
			localSetPath( value );
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processSrc_ );
		}

		static NMsgSrc* getReinitSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitSrc_ );
		}

		static NMsgSrc* getMolSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->molSrc_ );
		}

		static NMsgSrc* getBufSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->bufSrc_ );
		}

		static NMsgSrc* getSumTotSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->sumTotSrc_ );
		}

		static NMsgSrc* getProcessReacSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processReacSrc_ );
		}

		static NMsgSrc* getReinitReacSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitReacSrc_ );
		}

		static NMsgSrc* getProcessEnzSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processEnzSrc_ );
		}

		static NMsgSrc* getReinitEnzSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitEnzSrc_ );
		}

		static NMsgSrc* getProcessMmEnzSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processMmEnzSrc_ );
		}

		static NMsgSrc* getReinitMmEnzSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitMmEnzSrc_ );
		}

		static NMsgSrc* getProcessRateSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processRateSrc_ );
		}

		static NMsgSrc* getReinitRateSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitRateSrc_ );
		}

		static NMsgSrc* getRateSrc( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->rateSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void processFuncLocal( ProcInfo info ) {
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void reinitFuncLocal(  ) {
		}
		static void reinitFunc( Conn* c ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void molFuncLocal( double n, double nInit, int mode, long index );
		static void molFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				molFuncLocal( n, nInit, mode,
		static_cast< SolverConn* >( c )->index() );
		}

		void bufMolFuncLocal( double n, double nInit, int mode, long index );
		static void bufMolFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				bufMolFuncLocal( n, nInit, mode,
		static_cast< SolverConn* >( c )->index() );
		}

		void sumTotMolFuncLocal( double n, double nInit, int mode, long index );
		static void sumTotMolFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				sumTotMolFuncLocal( n, nInit, mode,
		static_cast< SolverConn* >( c )->index() );
		}

		void rateFuncLocal( double yPrime, long index ) {
		}
		static void rateFunc( Conn* c, double yPrime ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				rateFuncLocal( yPrime,
		static_cast< SolverConn* >( c )->index() );
		}

		void reacFuncLocal( double kf, double kb, long index ) {
			cout << "Ksolve::reacFuncLocal from index = " <<
					index << ", " << name() << 
				", kf= " << kf << ", kb= " << kb << "\n";
		}
		static void reacFunc( Conn* c, double kf, double kb ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				reacFuncLocal( kf, kb,
		static_cast< SolverConn* >( c )->index() );
		}

		void enzFuncLocal( double k1, double k2, double k3, long index ) {
			cout << "Ksolve::enzFuncLocal from index = " <<
					index << ", " << name() << 
				", k1= " << k1 << ", k2= " << k2 <<
				", k3= " << k3 << "\n";
		}
		static void enzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				enzFuncLocal( k1, k2, k3,
		static_cast< SolverConn* >( c )->index() );
		}

		void mmEnzFuncLocal( double k1, double k2, double k3, long index ) {
			cout << "Ksolve::mmEnzFuncLocal from index = " <<
					index << ", " << name() << 
				", k1= " << k1 << ", k2= " << k2 <<
				", k3= " << k3 << "\n";
		}
		static void mmEnzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				mmEnzFuncLocal( k1, k2, k3,
		static_cast< SolverConn* >( c )->index() );
		}

		void tabFuncLocal( double n, long index ) {
		}
		static void tabFunc( Conn* c, double n ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				tabFuncLocal( n,
		static_cast< SolverConn* >( c )->index() );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getMolSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->molSolveConn_ );
		}
		/*
		static Conn* getBufSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->bufSolveConn_ );
		}
		static Conn* getSumTotSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->sumTotSolveConn_ );
		}
		*/
		static Conn* getReacSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reacSolveConn_ );
		}
		static Conn* getEnzSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->enzSolveConn_ );
		}
		static Conn* getMmEnzSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->mmEnzSolveConn_ );
		}
		static Conn* getTabSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->tabSolveConn_ );
		}
		static Conn* getRateSolveConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->rateSolveConn_ );
		}
		static Conn* getProcessInConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->processInConn_ );
		}
		static Conn* getReinitInConn( Element* e ) {
			return &( static_cast< KsolveWrapper* >( e )->reinitInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Ksolve* p = dynamic_cast<const Ksolve *>(proto);
			// if (p)... and so on. 
			return new KsolveWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processSrc_;
		NMsgSrc0 reinitSrc_;
		NMsgSrc1< double > molSrc_;
		NMsgSrc1< double > bufSrc_;
		NMsgSrc1< double > sumTotSrc_;
		NMsgSrc1< ProcInfo > processReacSrc_;
		NMsgSrc0 reinitReacSrc_;
		NMsgSrc1< ProcInfo > processEnzSrc_;
		NMsgSrc0 reinitEnzSrc_;
		NMsgSrc1< ProcInfo > processMmEnzSrc_;
		NMsgSrc0 reinitMmEnzSrc_;
		NMsgSrc1< ProcInfo > processRateSrc_;
		NMsgSrc0 reinitRateSrc_;
		NMsgSrc1< double > rateSrc_;
		SolveMultiConn molSolveConn_;
		// SolveMultiConn bufSolveConn_;
		// SolveMultiConn sumTotSolveConn_;
		SolveMultiConn reacSolveConn_;
		SolveMultiConn enzSolveConn_;
		SolveMultiConn mmEnzSolveConn_;
		SolveMultiConn tabSolveConn_;
		SolveMultiConn rateSolveConn_;
		UniConn< processInConnKsolveLookup > processInConn_;
		UniConn< reinitInConnKsolveLookup > reinitInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void setPathLocal( const string& value );

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _KsolveWrapper_h
