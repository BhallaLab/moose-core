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

struct MethodInfo
{
	string description;
	bool isStochastic;
	bool isSpatial;
	bool isVariableDt;
	bool isImplicit;
	bool isSingleParticle;
	bool isMultiscale;
	// May need other info here as well
};

class KineticManager
{
	public:
		KineticManager();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		static void setAuto( const Conn& c, bool value );
		static bool getAuto( const Element* e );
		static void setStochastic( const Conn& c, bool value );
		static bool getStochastic( const Element* e );
		static void setSpatial( const Conn& c, bool value );
		static bool getSpatial( const Element* e );
		static void setMethod( const Conn& c, string value );
		static string getMethod( const Element* e );

		// Some readonly fields with more info about the methods.
		static bool getVariableDt( const Element* e );
		static bool getSingleParticle( const Element* e );
		static bool getMultiscale( const Element* e );
		static bool getImplicit( const Element* e );
		static string getDescription( const Element* e );
		static double getRecommendedDt( const Element* e );
		static void setEulerError( const Conn& c, double value );
		static double getEulerError( const Element* e );

		// static string getMethodList( const Element* e );
		//
		void innerSetMethod( Element* e, string value );
		void setupSolver( Element* e );
		void setupDt( Element* e, double dt );
		double estimateDt( Element* e, Element** elm, string& field, 
			double error );
		double findEnzSubPropensity( Element* e ) const;
		double findEnzPrdPropensity( Element* e ) const;
		double findReacPropensity( Element* e, bool isPrd ) const;
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void reinitFunc( const Conn& c, ProcInfo info );
		void reinitFuncLocal( Element* e );
		static void processFunc( const Conn& c, ProcInfo info );
		/*
		void processFuncLocal( Element* e, ProcInfo info );
		*/
		static void reschedFunc( const Conn& c );
		void reschedFuncLocal( Element* e );

 // static void addMethod( name, description,
 // 					isStochastic,isSpatial, 
 // 					isVariableDt, isImplicit,
 //						isSingleParticle, isMultiscale );
		static void addMethod( const string& name, 
			const string& description,
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
