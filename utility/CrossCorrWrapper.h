/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _CrossCorrWrapper_h
#define _CrossCorrWrapper_h
class CrossCorrWrapper: 
	public CrossCorr, public Neutral
{
	friend Element* processConnCrossCorrLookup( const Conn* );
	friend Element* aSpikeInConnCrossCorrLookup( const Conn* );
	friend Element* bSpikeInConnCrossCorrLookup( const Conn* );
	friend Element* printInConnCrossCorrLookup( const Conn* );
    public:
		CrossCorrWrapper(const string& n)
		:
			Neutral( n )
			// processConn uses a templated lookup function,
			// aSpikeInConn uses a templated lookup function,
			// bSpikeInConn uses a templated lookup function,
			// printInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setThreshold( Conn* c, double value ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->threshold_ = value;
		}
		static double getThreshold( const Element* e ) {
			return static_cast< const CrossCorrWrapper* >( e )->threshold_;
		}
		static void setBinCount( Conn* c, int value ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->binCount_ = value;
		}
		static int getBinCount( const Element* e ) {
			return static_cast< const CrossCorrWrapper* >( e )->binCount_;
		}
		static void setBinWidth( Conn* c, double value ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->binWidth_ = value;
		}
		static double getBinWidth( const Element* e ) {
			return static_cast< const CrossCorrWrapper* >( e )->binWidth_;
		}
		static void setASpikeCount( Conn* c, int value ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->aSpikeCount_ = value;
		}
		static int getASpikeCount( const Element* e ) {
			return static_cast< const CrossCorrWrapper* >( e )->aSpikeCount_;
		}
		static void setBSpikeCount( Conn* c, int value ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->bSpikeCount_ = value;
		}
		static int getBSpikeCount( const Element* e ) {
			return static_cast< const CrossCorrWrapper* >( e )->bSpikeCount_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void aSpikeFuncLocal( double amplitude, double cTime );
		static void aSpikeFunc( Conn* c, double amplitude, double cTime ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->
				aSpikeFuncLocal( amplitude, cTime );
		}

		void bSpikeFuncLocal( double amplitude, double cTime );
		static void bSpikeFunc( Conn* c, double amplitude, double cTime ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->
				bSpikeFuncLocal( amplitude, cTime );
		}

		void printFuncLocal( string fileName, int plotMode );
		static void printFunc( Conn* c, string fileName, int plotMode ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->
				printFuncLocal( fileName, plotMode );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info ) {
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< CrossCorrWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< CrossCorrWrapper* >( e )->processConn_ );
		}
		static Conn* getASpikeInConn( Element* e ) {
			return &( static_cast< CrossCorrWrapper* >( e )->aSpikeInConn_ );
		}
		static Conn* getBSpikeInConn( Element* e ) {
			return &( static_cast< CrossCorrWrapper* >( e )->bSpikeInConn_ );
		}
		static Conn* getPrintInConn( Element* e ) {
			return &( static_cast< CrossCorrWrapper* >( e )->printInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const CrossCorr* p = dynamic_cast<const CrossCorr *>(proto);
			// if (p)... and so on. 
			return new CrossCorrWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		UniConn< processConnCrossCorrLookup > processConn_;
		UniConn< aSpikeInConnCrossCorrLookup > aSpikeInConn_;
		UniConn< bSpikeInConnCrossCorrLookup > bSpikeInConn_;
		UniConn< printInConnCrossCorrLookup > printInConn_;

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
#endif // _CrossCorrWrapper_h
