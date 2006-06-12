/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ForwardEulerWrapper_h
#define _ForwardEulerWrapper_h
class ForwardEulerWrapper: 
	public ForwardEuler, public Neutral
{
	friend Element* integrateConnForwardEulerLookup( const Conn* );
	friend Element* processConnForwardEulerLookup( const Conn* );
    public:
		ForwardEulerWrapper(const string& n)
		:
			Neutral( n ),
			reinitSrc_( &integrateConn_ ),
			integrateSrc_( &integrateConn_ )
			// integrateConn uses a templated lookup function,
			// processConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static int getIsInitialized( const Element* e ) {
			return static_cast< const ForwardEulerWrapper* >( e )->isInitialized_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getReinitSrc( Element* e ) {
			return &( static_cast< ForwardEulerWrapper* >( e )->reinitSrc_ );
		}

		static SingleMsgSrc* getIntegrateSrc( Element* e ) {
			return &( static_cast< ForwardEulerWrapper* >( e )->integrateSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void allocateFuncLocal( vector< double >*  y );
		static void allocateFunc( Conn* c, vector< double >*  y ) {
			static_cast< ForwardEulerWrapper* >( c->parent() )->
				allocateFuncLocal( y );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< ForwardEulerWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void reinitFuncLocal(  ) {
			reinitSrc_.send();
		}
		static void reinitFunc( Conn* c ) {
			static_cast< ForwardEulerWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getIntegrateConn( Element* e ) {
			return &( static_cast< ForwardEulerWrapper* >( e )->integrateConn_ );
		}
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< ForwardEulerWrapper* >( e )->processConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const ForwardEuler* p = dynamic_cast<const ForwardEuler *>(proto);
			// if (p)... and so on. 
			return new ForwardEulerWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc0 reinitSrc_;
		SingleMsgSrc2< vector< double >* , double > integrateSrc_;
		UniConn< integrateConnForwardEulerLookup > integrateConn_;
		UniConn< processConnForwardEulerLookup > processConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _ForwardEulerWrapper_h
