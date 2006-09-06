/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _MoleculeWrapper_h
#define _MoleculeWrapper_h
class MoleculeWrapper: 
	public Molecule, public Neutral
{
	friend Element* processConnMoleculeLookup( const Conn* );
	// friend Element* solveConnMoleculeLookup( const Conn* );
	friend Element* sumProcessInConnMoleculeLookup( const Conn* );
    public:
		MoleculeWrapper(const string& n)
		:
			Neutral( n ),
			reacSrc_( &reacConn_ ),
			nSrc_( &nOutConn_ ),
			solveSrc_( &processConn_ ),
			// processConn uses a templated lookup function,
			// solveConn uses a templated lookup function,
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
		static void setMode( Conn* c, int value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->mode_ = value;
		}
		static int getMode( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->mode_;
		}
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		double localGetConc() const;
		static double getConc( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->
			localGetConc();
		}
		void localSetConc( double value );
		static void setConc( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
			localSetConc( value );
		}
		double localGetConcInit() const;
		static double getConcInit( const Element* e ) {
			return static_cast< const MoleculeWrapper* >( e )->
			localGetConcInit();
		}
		void localSetConcInit( double value );
		static void setConcInit( Conn* c, double value ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
			localSetConcInit( value );
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

		static SingleMsgSrc* getSolveSrc( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->solveSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reacFuncLocal( double A, double B ) {
			A_ += A;
			B_ += B;
		}
		static void reacFunc( Conn* c, double A, double B ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				reacFuncLocal( A, B );
		}

		void prdFuncLocal( double A, double B ) {
			A_ += A;
			B_ += B;
		}
		static void prdFunc( Conn* c, double A, double B ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				prdFuncLocal( A, B );
		}

		void sumTotalFuncLocal( double n ) {
			total_ += n;
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

		void solveFuncLocal( double n ) {
			n_ = n;
		}
		static void solveFunc( Conn* c, double n ) {
			static_cast< MoleculeWrapper* >( c->parent() )->
				solveFuncLocal( n );
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
		/*
		static Conn* getSolveConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->solveConn_ );
		}
		*/
		static Conn* getReacConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->reacConn_ );
		}
		static Conn* getNOutConn( Element* e ) {
			return &( static_cast< MoleculeWrapper* >( e )->nOutConn_ );
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
		SingleMsgSrc3< double, double, int > solveSrc_;
		UniConn< processConnMoleculeLookup > processConn_;
		// UniConn< solveConnMoleculeLookup > solveConn_;
		MultiConn reacConn_;
		MultiConn nOutConn_;
		PlainMultiConn prdInConn_;
		PlainMultiConn sumTotalInConn_;
		UniConn< sumProcessInConnMoleculeLookup > sumProcessInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void solverUpdate( const Finfo* f, SolverOp s ) const;
		bool isSolved( ) const;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _MoleculeWrapper_h
