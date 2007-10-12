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
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		/*
		static void reinitFunc( const Conn& c, ProcInfo info );
		void reinitFuncLocal( Element* e );
		static void processFunc( const Conn& c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );
		*/

	private:
		bool auto_;	// Default true. Pick method automatically
		bool stochastic_; // Default False
		bool spatial_; // Default False
		string method_; // R/W field, but if it is set explicitly the
						// auto_flag gets turned off. Default options:
						// RK5 for determ
						// Fast Gillespie 1 for stoch
						// yet to decide for spatial set.
};

#endif // _KINETIC_MANAGER_H
