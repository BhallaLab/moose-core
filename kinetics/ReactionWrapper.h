/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ReactionWrapper_h
#define _ReactionWrapper_h
class ReactionWrapper: 
	public Reaction, public Neutral
{
	friend Element* processConnReactionLookup( const Conn* );
	friend Element* solveConnReactionLookup( const Conn* );
    public:
		ReactionWrapper(const string& n)
		:
			Neutral( n ),
			subSrc_( &subConn_ ),
			prdSrc_( &prdConn_ ),
			solveSrc_( &solveConn_ ),
			// processConn uses a templated lookup function,
			// solveConn uses a templated lookup function,
			subConn_( this ),
			prdConn_( this ),
			scaleKfInConn_( this ),
			scaleKbInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setKf( Conn* c, double value ) {
			static_cast< ReactionWrapper* >( c->parent() )->kf_ = value;
		}
		static double getKf( const Element* e ) {
			return static_cast< const ReactionWrapper* >( e )->kf_;
		}
		static void setKb( Conn* c, double value ) {
			static_cast< ReactionWrapper* >( c->parent() )->kb_ = value;
		}
		static double getKb( const Element* e ) {
			return static_cast< const ReactionWrapper* >( e )->kb_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getSubSrc( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->subSrc_ );
		}

		static NMsgSrc* getPrdSrc( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->prdSrc_ );
		}

		static SingleMsgSrc* getSolveSrc( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->solveSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  ) {
			A_ = B_ = 0;
		}
		static void reinitFunc( Conn* c ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void subFuncLocal( double n ) {
			A_ *= n;
		}
		static void subFunc( Conn* c, double n ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				subFuncLocal( n );
		}

		void prdFuncLocal( double n ) {
			B_ *= n;
		}
		static void prdFunc( Conn* c, double n ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				prdFuncLocal( n );
		}

		void scaleKfFuncLocal( double k ) {
			A_ *= k;
		}
		static void scaleKfFunc( Conn* c, double k ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				scaleKfFuncLocal( k );
		}

		void scaleKbFuncLocal( double k ) {
			B_ *= k;
		}
		static void scaleKbFunc( Conn* c, double k ) {
			static_cast< ReactionWrapper* >( c->parent() )->
				scaleKbFuncLocal( k );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->processConn_ );
		}
		static Conn* getSolveConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->solveConn_ );
		}
		static Conn* getSubConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->subConn_ );
		}
		static Conn* getPrdConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->prdConn_ );
		}
		static Conn* getScaleKfInConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->scaleKfInConn_ );
		}
		static Conn* getScaleKbInConn( Element* e ) {
			return &( static_cast< ReactionWrapper* >( e )->scaleKbInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Reaction* p = dynamic_cast<const Reaction *>(proto);
			// if (p)... and so on. 
			return new ReactionWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc2< double, double > subSrc_;
		NMsgSrc2< double, double > prdSrc_;
		SingleMsgSrc2< double, double > solveSrc_;
		UniConn< processConnReactionLookup > processConn_;
		UniConn< solveConnReactionLookup > solveConn_;
		MultiConn subConn_;
		MultiConn prdConn_;
		PlainMultiConn scaleKfInConn_;
		PlainMultiConn scaleKbInConn_;

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
#endif // _ReactionWrapper_h
