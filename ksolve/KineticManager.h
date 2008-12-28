/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _KINETIC_MANAGER_H
#define _KINETIC_MANAGER_H

class KMethodInfo
{
	public:
		KMethodInfo()
		: isStochastic( 0 ), isSpatial( 0 ), isVariableDt( 0 ),
		isImplicit( 0 ), isSingleParticle( 0 ), isMultiscale( 0 ),
		description( "" )
		{;}

		KMethodInfo( bool stoch, bool spat, bool var, bool imp, bool sp, bool ms, const string& desc )
		: isStochastic( stoch ), isSpatial( spat ), isVariableDt( var ),
		isImplicit( imp ), isSingleParticle( sp ), isMultiscale( ms ),
		description( desc )
		{;}

		~KMethodInfo() {;}
		bool isStochastic;
		bool isSpatial;
		bool isVariableDt;
		bool isImplicit;
		bool isSingleParticle;
		bool isMultiscale;
		string description;
	// May need other info here as well
};

class KineticManager: public KinCompt
{
	public:
		KineticManager();
		virtual ~KineticManager()
		{ ; }
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		static void setAuto( const Conn* c, bool value );
		static bool getAuto( Eref e );
		static void setStochastic( const Conn* c, bool value );
		static bool getStochastic( Eref e );
		static void setSpatial( const Conn* c, bool value );
		static bool getSpatial( Eref e );
		static void setMethod( const Conn* c, string value );
		static string getMethod( Eref e );

		// Some readonly fields with more info about the methods.
		static bool getVariableDt( Eref e );
		static bool getSingleParticle( Eref e );
		static bool getMultiscale( Eref e );
		static bool getImplicit( Eref e );
		static string getDescription( Eref e );
		static double getRecommendedDt( Eref e );
		static void setEulerError( const Conn* c, double value );
		static double getEulerError( Eref e );

		static double getLoadEstimate( Eref e );
		static unsigned int getMemEstimate( Eref e );

		// static string getMethodList( Eref e );
		//
		void innerSetMethod( Eref e, string value );
		void setupSolver( Eref e );
		void setupDt( Eref e, double dt );

		/**
		 * Estimates timestep and other aspects of computational load.
		 * Uses an Euler error calculation to assign a recommended
		 * Dt for a given error value. Different numerical methods will
		 * typically be a certain factor better than this.
		 * In addition, uses rules of thumb to estimate load for other 
		 * numerical methods.
		 */
		double estimateDt( Id mgr, Id& elm, string& field, 
			double error, string method ) ;
		double findEnzSubPropensity( Eref e ) const;
		double findEnzPrdPropensity( Eref e ) const;
		double findReacPropensity( Eref e, bool isPrd ) const;
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void reinitFunc( const Conn* c, ProcInfo info );
		void reinitFuncLocal( Eref e );
		static void processFunc( const Conn* c, ProcInfo info );
		static void reschedFunc( const Conn* c );
		void reschedFuncLocal( Eref e );
		static void estimateDtFunc( const Conn* c, string method );

		/**
 		* innerSetSize is specialized from KinCompt because when 
		* rescaling is done, the stoich must modify rates, n and nInit,
		* in addition to the modifications on the original objects.
 		*/
		void innerSetSize( Eref e, double value, bool ignoreRescale );

 // static void addMethod( name, description,
 // 					isStochastic,isSpatial, 
 // 					isVariableDt, isImplicit,
 //						isSingleParticle, isMultiscale );
		static void addMethod( const char* name, 
			const char* description,
			bool isStochastic, bool isSpatial, 
			bool isVariableDt, bool isImplicit,
			bool isSingleParticle, bool isMultiscale );
		///////////////////////////////////////////////////
		// Utility functions.
		///////////////////////////////////////////////////
		/**
 		* Finds descendants of this KineticManager and puts into the
 		* ret vector. Returns the # of descendants found. 
 		* If it encounters another KineticManager among descendants, it
 		* bypasses it and its children, unless the child KineticManager
 		* has the 'neutral' or 'ee' method.
 		* Goal is to build the path of solved elements, but allow
		* subsidiary KineticManagers to do their own thing.
 		*/
		//unsigned int findDescendants( Eref e, vector< Id >& ret );

	private:
		bool auto_;	// Default true. Pick method automatically
		bool stochastic_; // Default False
		bool spatial_; // Default False
		string method_; // R/W field, but if it is set explicitly the
						// auto_flag gets turned off. Default options:
						// RK5 for determ
						// Fast Gillespie 1 for stoch
						// yet to decide for spatial set.
		bool implicit_;	// default False
		bool variableDt_; // Default False
		bool multiscale_; // Default False
		bool singleParticle_; // Default False
		string description_;
		double recommendedDt_;

		/**
		 * Estimate of computational load of model
		 * Expressed roughly as # of flops per second sim time
		 */
		double loadEstimate_;
		unsigned int memEstimate_; // Est # of bytes for building model
		double eulerError_;
		string lastMethodForEstimateDt_;
		static const double RKload;
		static const double GillespieLoad;
		static const unsigned int RKmemLoad;
		static const unsigned int GillespieMemLoad;
		static const unsigned int elementMemLoad;
};

const Cinfo* initKineticManagerCinfo();

#endif // _KINETIC_MANAGER_H
