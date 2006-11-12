/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MultiSiteWrapper_h
#define _MultiSiteWrapper_h
class MultiSiteWrapper: 
	public MultiSite, public Neutral
{
	friend Element* processConnMultiSiteLookup( const Conn* );
	friend Element* solveConnMultiSiteLookup( const Conn* );
    public:
		MultiSiteWrapper(const string& n)
		:
			Neutral( n ),
			scaleSrc_( &scaleOutConn_ ),
			solveSrc_( &solveConn_ ),
			// processConn uses a templated lookup function,
			// solveConn uses a templated lookup function,
			scaleOutConn_( this ),
			siteInConn_( this )
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setNTotal( Conn* c, double value ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->nTotal_ = value;
		}
		static double getNTotal( const Element* e ) {
			return static_cast< const MultiSiteWrapper* >( e )->nTotal_;
		}
		static void setStates(
			Element* e, unsigned long index, int value );
		static int getStates(
			const Element* e, unsigned long index );
		static void setOccupancy(
			Element* e, unsigned long index, double value );
		static double getOccupancy(
			const Element* e, unsigned long index );
		static void setRates(
			Element* e, unsigned long index, double value );
		static double getRates(
			const Element* e, unsigned long index );
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		int localGetNSites() const;
		static int getNSites( const Element* e ) {
			return static_cast< const MultiSiteWrapper* >( e )->
			localGetNSites();
		}
		void localSetNSites( int value );
		static void setNSites( Conn* c, int value ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
			localSetNSites( value );
		}
		int localGetNStates() const;
		static int getNStates( const Element* e ) {
			return static_cast< const MultiSiteWrapper* >( e )->
			localGetNStates();
		}
		void localSetNStates( int value );
		static void setNStates( Conn* c, int value ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
			localSetNStates( value );
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static NMsgSrc* getScaleSrc( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->scaleSrc_ );
		}

		static SingleMsgSrc* getSolveSrc( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->solveSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void siteFuncLocal( double n, long index ) {
			if ( index < static_cast< long >( fraction_.size() ) )
				fraction_[index] = n / nTotal_;
		}
		static void siteFunc( Conn* c, double n ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
				siteFuncLocal( n,
				static_cast< SolverConn* >( c )->index() );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void solveFuncLocal(  ) {
		}
		static void solveFunc( Conn* c ) {
			static_cast< MultiSiteWrapper* >( c->parent() )->
				solveFuncLocal(  );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->processConn_ );
		}
		static Conn* getSolveConn( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->solveConn_ );
		}
		static Conn* getScaleOutConn( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->scaleOutConn_ );
		}
		static Conn* getSiteInConn( Element* e ) {
			return &( static_cast< MultiSiteWrapper* >( e )->siteInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const MultiSite* p = dynamic_cast<const MultiSite *>(proto);
			// if (p)... and so on. 
			return new MultiSiteWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		NMsgSrc1< double > scaleSrc_;
		SingleMsgSrc3< const vector< int >*, const vector< double >*, int > solveSrc_;
		UniConn< processConnMultiSiteLookup > processConn_;
		UniConn< solveConnMultiSiteLookup > solveConn_;
		MultiConn scaleOutConn_;
		SolveMultiConn siteInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void solverUpdate( const Finfo* f, SolverOp s ) const;
		bool isSolved() const;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _MultiSiteWrapper_h
