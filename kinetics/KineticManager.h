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

class KineticManager
{
	public:
		KineticManager();
		
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

		// static string getMethodList( Eref e );
		//
		void innerSetMethod( Eref e, string value );
		void setupSolver( Eref e );
		void setupDt( Eref e, double dt );
		double estimateDt( Id mgr, Id& elm, string& field, double error ) ;
		double findEnzSubPropensity( Eref e ) const;
		double findEnzPrdPropensity( Eref e ) const;
		double findReacPropensity( Eref e, bool isPrd ) const;
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void reinitFunc( const Conn* c, ProcInfo info );
		void reinitFuncLocal( Eref e );
		static void processFunc( const Conn* c, ProcInfo info );
		/*
		void processFuncLocal( Element* e, ProcInfo info );
		*/
		static void reschedFunc( const Conn* c );
		void reschedFuncLocal( Eref e );

 // static void addMethod( name, description,
 // 					isStochastic,isSpatial, 
 // 					isVariableDt, isImplicit,
 //						isSingleParticle, isMultiscale );
		static void addMethod( const char* name, 
			const char* description,
			bool isStochastic, bool isSpatial, 
			bool isVariableDt, bool isImplicit,
			bool isSingleParticle, bool isMultiscale );

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
		double eulerError_;
};

#endif // _KINETIC_MANAGER_H
