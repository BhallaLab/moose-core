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
			rateSrc_( &rateSolveConn_ ),
			molSolveConn_( this ),
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

		void molFuncLocal( double n, double nInit, int mode ) {
			cout << "Got msg from mol: " << n << ", " << nInit <<
				", " << mode << "\n";
		}

		static void molFunc( Conn* c, double n, double nInit, int mode ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				molFuncLocal( n, nInit, mode );
		}

		void rateFuncLocal( double yPrime ) {
		}
		static void rateFunc( Conn* c, double yPrime ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				rateFuncLocal( yPrime );
		}

		void reacFuncLocal( double kf, double kb ) {
		}
		static void reacFunc( Conn* c, double kf, double kb ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				reacFuncLocal( kf, kb );
		}

		void enzFuncLocal( double k1, double k2, double k3 ) {
		}
		static void enzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				enzFuncLocal( k1, k2, k3 );
		}

		void mmEnzFuncLocal( double k1, double k2, double k3 ) {
		}
		static void mmEnzFunc( Conn* c, double k1, double k2, double k3 ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				mmEnzFuncLocal( k1, k2, k3 );
		}

		void tabFuncLocal( double n ) {
		}
		static void tabFunc( Conn* c, double n ) {
			static_cast< KsolveWrapper* >( c->parent() )->
				tabFuncLocal( n );
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
		NMsgSrc1< double > rateSrc_;
		MultiConn molSolveConn_;
		MultiConn reacSolveConn_;
		MultiConn enzSolveConn_;
		MultiConn mmEnzSolveConn_;
		MultiConn tabSolveConn_;
		MultiConn rateSolveConn_;
		UniConn< processInConnKsolveLookup > processInConn_;
		UniConn< reinitInConnKsolveLookup > reinitInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void setPathLocal( const string& value );
		void molZombify( Element* e, Field& solveSrc );

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _KsolveWrapper_h
