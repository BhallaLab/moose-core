/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _StateScanner_h
#define _StateScanner_h
class StateScanner
{
#ifdef DO_UNIT_TESTS
	friend void testStateScanner();
#endif
	public:
		StateScanner();
		~StateScanner();
		
		///////////////////////////////////////////////////
		// Field function definitions
		///////////////////////////////////////////////////
		static double getSettleTime( Eref e );
		static void setSettleTime( const Conn* c, double value );
		static double getSolutionSeparation( Eref e );
		static void setSolutionSeparation( const Conn* c, double value );
		static unsigned int getStateCategory(Eref e, const unsigned int& i);
		unsigned int localGetStateCategory( unsigned int i) const;
		static void setStateCategory( 
			const Conn* c, unsigned int val, const unsigned int& i );
		static bool getUseLog( Eref e );
		static void setUseLog( const Conn* c, bool value );
		static bool getUseSS( Eref e );
		static void setUseSS( const Conn* c, bool value );
		static bool getUseRisingDose( Eref e );
		static void setUseRisingDose( const Conn* c, bool value );
		static bool getUseBufferDose( Eref e );
		static void setUseBufferDose( const Conn* c, bool value );
		/*
		static unsigned int getNumTrackedMolecules( Eref e );
		static void setNumTrackedMolecules( const Conn* c, unsigned int value );
		void localSetNumTrackedMolecules( unsigned int value );
		static Id getTrackedMolecule( Eref e, const unsigned int& i );
		static void setTrackedMolecule(
			const Conn* c, Id val, const unsigned int& i );
		Id localGetTrackedMolecule( unsigned int i ) const;
		void localSetTrackedMolecule( Id elm, unsigned int i );
		*/
		static unsigned int getClassification( Eref e );

		///////////////////////////////////////////////////
		// Msg Dest function definitions
		///////////////////////////////////////////////////
		//
		static void addTrackedMolecule( const Conn* c, Id val );
		static void dropTrackedMolecule( const Conn* c, Id val );
		
		static void doseResponse( const Conn* c, 
			Id variableMol, 
			double start, double end, 
			unsigned int numSteps );

		static void logDoseResponse( const Conn* c, 
			Id variableMol, 
			double start, double end, 
			unsigned int numSteps );

		void innerDoseResponse( Eref me, Id variableMol, 
			double start, double end, 
			unsigned int numSteps,
			bool useLog );

		static void classifyStates( const Conn* c, 
			unsigned int numStartingPoints, bool useMonteCarlo);

		static void saveAsCSV( const Conn* c, string fname );
		static void saveAsXplot( const Conn* c, string fname );


		///////////////////////////////////////////////////
		// Utility functions for doing doser
		///////////////////////////////////////////////////
		bool initDoser( 
			double start, double end, unsigned int numSteps, bool useLog);
		bool advanceDoser();
		void setControlParameter( Id& variableMol );
		void settle( Eref me, Id& cj, Id& ss );
		void makeChildTable( Eref me, string name );
		
	private:
		bool isMoleculeIndexGood( unsigned int i ) const;

		///////////////////////////////////////////////////
		// Utility functions for classifying states
		///////////////////////////////////////////////////
		void innerClassifyStates(
			Eref me,
			unsigned int numStartingPoints,
			bool useMonteCarlo
			);
		bool stateSettle( Eref me, Id& cj, Id& ss );
		void checkIfUniqueState( Eref me );
		void classify();
		
		///////////////////////////////////////////////////
		// Internal fields.
		///////////////////////////////////////////////////
		double settleTime_;
		double solutionSeparation_;
		unsigned int numTrackedMolecules_;
		vector< Id > trackedMolecule_;
		vector< unsigned int> stateCategories_;

		unsigned int numSolutions_;
		unsigned int numStable_;
		unsigned int numSaddle_;
		unsigned int numOsc_;
		unsigned int numOther_;
		unsigned int classification_;
		bool useLog_; /// Use logarithmic increments in dose-response
		bool useRisingDose_; /// Do a rising series.
		bool useBufferDose_; /// Use buffering in dose-response.
		bool useReinit_; /// Reinit each cycle if in buffering mode.
		bool useSS_; /// Use the SteadyState solver, rather than time-series
		double x_; /// current value in dose-response
		double dx_; /// increment: summed or multiplied, in dose-response
		double lastx_; /// previous value in dose-response
		double end_; /// terminating value in dose-response
		static const double EPSILON;
};

extern const Cinfo* initStateScannerCinfo();
#endif // _StateScanner_h
