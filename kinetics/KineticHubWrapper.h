/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _KineticHubWrapper_h
#define _KineticHubWrapper_h
class KineticHubWrapper: 
	public KineticHub, public Neutral
{
	friend Element* hubConnKineticHubLookup( const Conn* );
	friend Element* updateOutConnKineticHubLookup( const Conn* );
	friend Element* processInConnKineticHubLookup( const Conn* );
	friend Element* reinitInConnKineticHubLookup( const Conn* );
    public:
		KineticHubWrapper(const string& n)
		:
			Neutral( n ),
			processMolSrc_( &molSolveConn_ ),
			reinitMolSrc_( &molSolveConn_ ),
			molSrc_( &molSolveConn_ ),
			bufSrc_( &bufSolveConn_ ),
			processBufSrc_( &bufSolveConn_ ),
			reinitBufSrc_( &bufSolveConn_ ),
			sumTotSrc_( &sumTotSolveConn_ ),
			processSumTotSrc_( &sumTotSolveConn_ ),
			reinitSumTotSrc_( &reinitSumTotOutConn_ ),
			processReacSrc_( &reacSolveConn_ ),
			reinitReacSrc_( &reacSolveConn_ ),
			processEnzSrc_( &enzSolveConn_ ),
			reinitEnzSrc_( &enzSolveConn_ ),
			processMmEnzSrc_( &mmEnzSolveConn_ ),
			reinitMmEnzSrc_( &mmEnzSolveConn_ ),
			processRateSrc_( &rateSolveConn_ ),
			reinitRateSrc_( &rateSolveConn_ ),
			rateSrc_( &rateSolveConn_ ),
			updateSrc_( &updateOutConn_ ),
			molSolveConn_( this ),
			bufSolveConn_( this ),
			sumTotSolveConn_( this ),
			reacSolveConn_( this ),
			enzSolveConn_( this ),
			mmEnzSolveConn_( this ),
			tabSolveConn_( this ),
			rateSolveConn_( this ),
			// hubConn uses a templated lookup function,
			reinitSumTotOutConn_( this ),
			// updateOutConn uses a templated lookup function,
			// processInConn uses a templated lookup function,
			// reinitInConn uses a templated lookup function,
			sumTotInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getProcessMolSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processMolSrc_ );
		}

		static NMsgSrc* getReinitMolSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitMolSrc_ );
		}

		static NMsgSrc* getMolSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->molSrc_ );
		}

		static NMsgSrc* getBufSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->bufSrc_ );
		}

		static NMsgSrc* getProcessBufSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processBufSrc_ );
		}

		static NMsgSrc* getReinitBufSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitBufSrc_ );
		}

		static NMsgSrc* getSumTotSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->sumTotSrc_ );
		}

		static NMsgSrc* getProcessSumTotSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processSumTotSrc_ );
		}

		static NMsgSrc* getReinitSumTotSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitSumTotSrc_ );
		}

		static NMsgSrc* getProcessReacSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processReacSrc_ );
		}

		static NMsgSrc* getReinitReacSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitReacSrc_ );
		}

		static NMsgSrc* getProcessEnzSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processEnzSrc_ );
		}

		static NMsgSrc* getReinitEnzSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitEnzSrc_ );
		}

		static NMsgSrc* getProcessMmEnzSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processMmEnzSrc_ );
		}

		static NMsgSrc* getReinitMmEnzSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitMmEnzSrc_ );
		}

		static NMsgSrc* getProcessRateSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processRateSrc_ );
		}

		static NMsgSrc* getReinitRateSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitRateSrc_ );
		}

		static NMsgSrc* getRateSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->rateSrc_ );
		}

		static SingleMsgSrc* getUpdateSrc( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->updateSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void processFuncLocal( ProcInfo info ) {
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void reinitFuncLocal(  ) {
		}
		static void reinitFunc( Conn* c ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void molSizesFuncLocal( int nMol, int nBuf, int nSumTot );
		static void molSizesFunc( Conn* c, int nMol, int nBuf, int nSumTot ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				molSizesFuncLocal( nMol, nBuf, nSumTot );
		}

		void rateSizesFuncLocal( int nReac, int nEnz, int nMmEnz );
		static void rateSizesFunc( Conn* c, int nReac, int nEnz, int nMmEnz ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				rateSizesFuncLocal( nReac, nEnz, nMmEnz );
		}

		void molConnectionsFuncLocal( vector< double >*  S, vector< double >*  Sinit, vector< Element *>*  elist );
		static void molConnectionsFunc( Conn* c, vector< double >*  S, vector< double >*  Sinit, vector< Element *>*  elist ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				molConnectionsFuncLocal( S, Sinit, elist );
		}

		void rateTermInfoFuncLocal( vector< RateTerm* >*  rates, int useOneWayReacs ) {
			rates_ = rates;
			useOneWayReacs_ = useOneWayReacs;
		}
		static void rateTermInfoFunc( Conn* c, vector< RateTerm* >*  rates, int useOneWayReacs ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				rateTermInfoFuncLocal( rates, useOneWayReacs );
		}

		void reacConnectionFuncLocal( int rateTermIndex, Element* reac );
		static void reacConnectionFunc( Conn* c, int rateTermIndex, Element* reac ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				reacConnectionFuncLocal( rateTermIndex, reac );
		}

		void enzConnectionFuncLocal( int rateTermIndex, Element* enz );
		static void enzConnectionFunc( Conn* c, int rateTermIndex, Element* enz ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				enzConnectionFuncLocal( rateTermIndex, enz );
		}

		void mmEnzConnectionFuncLocal( int rateTermIndex, Element* enz );
		static void mmEnzConnectionFunc( Conn* c, int rateTermIndex, Element* enz ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				mmEnzConnectionFuncLocal( rateTermIndex, enz );
		}

		void molFuncLocal( double n, double nInit, int mode, long index );
		static void molFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				molFuncLocal( n, nInit, mode,
				static_cast< SolverConn* >( c )->index() );
		}

		void bufFuncLocal( double n, double nInit, int mode, long index );
		static void bufFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				bufFuncLocal( n, nInit, mode,
				static_cast< SolverConn* >( c )->index() );
		}

		void sumTotFuncLocal( double n, double nInit, int mode, long index );
		static void sumTotFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				sumTotFuncLocal( n, nInit, mode,
				static_cast< SolverConn* >( c )->index() );
		}

		void rateFuncLocal( double yPrime, long index ) {
		}
		static void rateFunc( Conn* c, double yPrime ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				rateFuncLocal( yPrime,
				static_cast< SolverConn* >( c )->index() );
		}

		void reacFuncLocal( double kf, double kb, long index );
		static void reacFunc( Conn* c, double kf, double kb ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				reacFuncLocal( kf, kb,
				static_cast< SolverConn* >( c )->index() );
		}

		void enzFuncLocal( double k1, double k2, double k3, long index );
		static void enzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				enzFuncLocal( k1, k2, k3,
				static_cast< SolverConn* >( c )->index() );
		}

		void mmEnzFuncLocal( double k1, double k2, double k3, long index );
		static void mmEnzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
				mmEnzFuncLocal( k1, k2, k3,
				static_cast< SolverConn* >( c )->index() );
		}

		void tabFuncLocal( double n, long index ) {
		}
		static void tabFunc( Conn* c, double n ) {
			static_cast< KineticHubWrapper* >( c->parent() )->
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
			return &( static_cast< KineticHubWrapper* >( e )->molSolveConn_ );
		}
		static Conn* getBufSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->bufSolveConn_ );
		}
		static Conn* getSumTotSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->sumTotSolveConn_ );
		}
		static Conn* getReacSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reacSolveConn_ );
		}
		static Conn* getEnzSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->enzSolveConn_ );
		}
		static Conn* getMmEnzSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->mmEnzSolveConn_ );
		}
		static Conn* getTabSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->tabSolveConn_ );
		}
		static Conn* getRateSolveConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->rateSolveConn_ );
		}
		static Conn* getHubConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->hubConn_ );
		}
		static Conn* getReinitSumTotOutConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitSumTotOutConn_ );
		}
		static Conn* getUpdateOutConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->updateOutConn_ );
		}
		static Conn* getProcessInConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->processInConn_ );
		}
		static Conn* getReinitInConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->reinitInConn_ );
		}
		static Conn* getSumTotInConn( Element* e ) {
			return &( static_cast< KineticHubWrapper* >( e )->sumTotInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const KineticHub* p = dynamic_cast<const KineticHub *>(proto);
			// if (p)... and so on. 
			return new KineticHubWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< ProcInfo > processMolSrc_;
		NMsgSrc0 reinitMolSrc_;
		NMsgSrc1< double > molSrc_;
		NMsgSrc1< double > bufSrc_;
		NMsgSrc1< ProcInfo > processBufSrc_;
		NMsgSrc1< ProcInfo > reinitBufSrc_;
		NMsgSrc1< double > sumTotSrc_;
		NMsgSrc1< ProcInfo > processSumTotSrc_;
		NMsgSrc1< ProcInfo > reinitSumTotSrc_;
		NMsgSrc1< ProcInfo > processReacSrc_;
		NMsgSrc0 reinitReacSrc_;
		NMsgSrc1< ProcInfo > processEnzSrc_;
		NMsgSrc0 reinitEnzSrc_;
		NMsgSrc1< ProcInfo > processMmEnzSrc_;
		NMsgSrc0 reinitMmEnzSrc_;
		NMsgSrc1< ProcInfo > processRateSrc_;
		NMsgSrc0 reinitRateSrc_;
		NMsgSrc1< double > rateSrc_;
		SingleMsgSrc0 updateSrc_;
		SolveMultiConn molSolveConn_;
		SolveMultiConn bufSolveConn_;
		SolveMultiConn sumTotSolveConn_;
		SolveMultiConn reacSolveConn_;
		SolveMultiConn enzSolveConn_;
		SolveMultiConn mmEnzSolveConn_;
		SolveMultiConn tabSolveConn_;
		SolveMultiConn rateSolveConn_;
		UniConn< hubConnKineticHubLookup > hubConn_;
		SolveMultiConn reinitSumTotOutConn_;
		UniConn< updateOutConnKineticHubLookup > updateOutConn_;
		UniConn< processInConnKineticHubLookup > processInConn_;
		UniConn< reinitInConnKineticHubLookup > reinitInConn_;
		SolveMultiConn sumTotInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void zombify( Element* e, Field& solveSrc );
		vector< int > reacIndex_;	
		vector< int > enzIndex_;
		vector< int > mmEnzIndex_;
		vector< RateTerm* >* rates_;
		int useOneWayReacs_;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _KineticHubWrapper_h
