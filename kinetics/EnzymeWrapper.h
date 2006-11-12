/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

// We need to manually initialize sk1 to 1.0, till mpp is fixed.
#ifndef _EnzymeWrapper_h
#define _EnzymeWrapper_h
class EnzymeWrapper: 
	public Enzyme, public Neutral
{
	friend Element* processConnEnzymeLookup( const Conn* );
//	friend Element* solveConnEnzymeLookup( const Conn* );
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
			solveSrc_( &processConn_ ),
			// processConn uses a templated lookup function,
			// solveConn uses a templated lookup function,
			// enzConn uses a templated lookup function,
			// cplxConn uses a templated lookup function,
			subConn_( this ),
			prdOutConn_( this ),
			// intramolInConn uses a templated lookup function,
			scaleKmInConn_( this ),
			scaleKcatInConn_( this )
		{
			;
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
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		double localGetKm() const;
		static double getKm( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->
			localGetKm();
		}
		void localSetKm( double value );
		static void setKm( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
			localSetKm( value );
		}
		double localGetKcat() const;
		static double getKcat( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->
			localGetKcat();
		}
		void localSetKcat( double value );
		static void setKcat( Conn* c, double value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
			localSetKcat( value );
		}
		int localGetMode() const;
		static int getMode( const Element* e ) {
			return static_cast< const EnzymeWrapper* >( e )->
			localGetMode();
		}
		void localSetMode( int value );
		static void setMode( Conn* c, int value ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
			localSetMode( value );
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

		static SingleMsgSrc* getSolveSrc( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->solveSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  ) {
			eA_ = sA_ = pA_ = B_ = e_ = 0.0;
			s_ = 1.0;
		}
		static void reinitFunc( Conn* c ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info ) {
			(this->*procFunc_)();
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void enzFuncLocal( double n ) {
			e_ = n;
		}
		static void enzFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				enzFuncLocal( n );
		}

		void cplxFuncLocal( double n ) {
			sA_ *= n;
			pA_ *= n;
		}
		static void cplxFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				cplxFuncLocal( n );
		}

		void subFuncLocal( double n ) {
			s_ *= n;
		}
		static void subFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				subFuncLocal( n );
		}

		void intramolFuncLocal( double n );
		static void intramolFunc( Conn* c, double n ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				intramolFuncLocal( n );
		}

		void scaleKmFuncLocal( double k );
		static void scaleKmFunc( Conn* c, double k ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				scaleKmFuncLocal( k );
		}

		void scaleKcatFuncLocal( double k ) {
			pA_ *= k;
		}
		static void scaleKcatFunc( Conn* c, double k ) {
			static_cast< EnzymeWrapper* >( c->parent() )->
				scaleKcatFuncLocal( k );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->processConn_ );
		}
		/*
		static Conn* getSolveConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->solveConn_ );
		}
		*/
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
		static Conn* getScaleKmInConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->scaleKmInConn_ );
		}
		static Conn* getScaleKcatInConn( Element* e ) {
			return &( static_cast< EnzymeWrapper* >( e )->scaleKcatInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Enzyme* p = dynamic_cast<const Enzyme *>(proto);
			// if (p)... and so on. 
			return new EnzymeWrapper(name);
		}

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
		SingleMsgSrc3< double, double, double > solveSrc_;
		UniConn< processConnEnzymeLookup > processConn_;
		// UniConn< solveConnEnzymeLookup > solveConn_;
		UniConn< enzConnEnzymeLookup > enzConn_;
		UniConn< cplxConnEnzymeLookup > cplxConn_;
		MultiConn subConn_;
		MultiConn prdOutConn_;
		UniConn< intramolInConnEnzymeLookup > intramolInConn_;
		PlainMultiConn scaleKmInConn_;
		PlainMultiConn scaleKcatInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void ( EnzymeWrapper::*procFunc_ )( );
		void implicitProcFunc();
		void explicitProcFunc();
		int innerGetMode() const {
			return ( procFunc_ == &EnzymeWrapper::implicitProcFunc );
		}
		void innerSetMode( int mode );
		void makeComplex();
		bool isSolved() const ;
		void solverUpdate( const Finfo* f, SolverOp s ) const;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _EnzymeWrapper_h
