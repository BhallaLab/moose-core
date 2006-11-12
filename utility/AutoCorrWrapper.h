/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2006 Niraj Dudani and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _AutoCorrWrapper_h
#define _AutoCorrWrapper_h
class AutoCorrWrapper: 
	public AutoCorr, public Neutral
{
	friend Element* processConnAutoCorrLookup( const Conn* );
	friend Element* spikeInConnAutoCorrLookup( const Conn* );
	friend Element* printInConnAutoCorrLookup( const Conn* );
    public:
		AutoCorrWrapper(const string& n)
		:
			Neutral( n )
			// processConn uses a templated lookup function,
			// spikeInConn uses a templated lookup function,
			// printInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setThreshold( Conn* c, double value ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->threshold_ = value;
		}
		static double getThreshold( const Element* e ) {
			return static_cast< const AutoCorrWrapper* >( e )->threshold_;
		}
		static void setBinCount( Conn* c, int value ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->binCount_ = value;
		}
		static int getBinCount( const Element* e ) {
			return static_cast< const AutoCorrWrapper* >( e )->binCount_;
		}
		static void setBinWidth( Conn* c, double value ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->binWidth_ = value;
		}
		static double getBinWidth( const Element* e ) {
			return static_cast< const AutoCorrWrapper* >( e )->binWidth_;
		}
		static void setSpikeCount( Conn* c, int value ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->spikeCount_ = value;
		}
		static int getSpikeCount( const Element* e ) {
			return static_cast< const AutoCorrWrapper* >( e )->spikeCount_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void spikeFuncLocal( double amplitude, double cTime );
		static void spikeFunc( Conn* c, double amplitude, double cTime ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->
				spikeFuncLocal( amplitude, cTime );
		}

		void printFuncLocal( string fileName, int plotMode );
		static void printFunc( Conn* c, string fileName, int plotMode ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->
				printFuncLocal( fileName, plotMode );
		}

		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info ) {
		}
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< AutoCorrWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< AutoCorrWrapper* >( e )->processConn_ );
		}
		static Conn* getSpikeInConn( Element* e ) {
			return &( static_cast< AutoCorrWrapper* >( e )->spikeInConn_ );
		}
		static Conn* getPrintInConn( Element* e ) {
			return &( static_cast< AutoCorrWrapper* >( e )->printInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const AutoCorr* p = dynamic_cast<const AutoCorr *>(proto);
			// if (p)... and so on. 
			return new AutoCorrWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		UniConn< processConnAutoCorrLookup > processConn_;
		UniConn< spikeInConnAutoCorrLookup > spikeInConn_;
		UniConn< printInConnAutoCorrLookup > printInConn_;

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
#endif // _AutoCorrWrapper_h
