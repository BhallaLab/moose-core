/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SIGNEUR_H
#define _SIGNEUR_H

class SigNeur
{
	public:
		SigNeur();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		// Object selection
		static void setCell( const Conn* c, Id value );
		static Id getCell( Eref e );
		static void setSpine( const Conn* c, Id value );
		static Id getSpine( Eref e );
		static void setDend( const Conn* c, Id value );
		static Id getDend( Eref e );
		static void setSoma( const Conn* c, Id value );
		static Id getSoma( Eref e );
		
		// Numerical Method selection for electrical cell model.
		static void setCellMethod( const Conn* c, string value );
		static string getCellMethod( Eref e );
		static void setSpineMethod( const Conn* c, string value );
		static string getSpineMethod( Eref e );
		static void setDendMethod( const Conn* c, string value );
		static string getDendMethod( Eref e );
		static void setSomaMethod( const Conn* c, string value );
		static string getSomaMethod( Eref e );

		/// Diffusion constant scaling factor, to globally modify diffusion
		static void setDscale( const Conn* c, double value );
		static double getDscale( Eref e );

		/// Options for parallel configuration. 
		static void setParallelMode( const Conn* c, int value );
		static int getParallelMode( Eref e );

		/// Timestep at which to exchange data between elec and sig models.
		static void setUpdateStep( const Conn* c, double value );
		static double getUpdateStep( Eref e );

		/// Map between electrical and signaling channel representations
		static void setChannelMap( const Conn* c, 
			string val, const string& i );
		static string getChannelMap( Eref e, const string& i );

		/// Map between electrical and signaling Calcium representations
		static void setCalciumMap( const Conn* c, 
			string val, const string& i );
		static string getCalciumMap( Eref e, const string& i );

		/// Scale factors for calcium conversion between models.
		static void setCalciumScale( const Conn* c, double value );
		static double getCalciumScale( Eref e );
		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void build( const Conn* c );
		void innerBuild( const Conn* c );

		///////////////////////////////////////////////////
		// Setup function definitions
		///////////////////////////////////////////////////
		bool traverseCell();

	private:
		Id cell_; /// Prototype cell electrical model
		Id spine_; /// Prototype spine signaling model
		Id dend_; /// prototype dendrite signaling model
		Id soma_; /// prototype soma signaling model
		string cellMethod_;
		string spineMethod_;
		string dendMethod_;
		string somaMethod_;
		double Dscale_;	/// Diffusion scale factor.
		int parallelMode_; /// Later redo to something more friendly
		double updateStep_;
		map< string, string > channelMap_; /// Channel name mapping
			// Scale factors are worked out from gmax and CoInit.
			
		map< string, string > calciumMap_; /// Calcium pool name mapping
		double calciumScale_;	/// Scaling factor between elec and sig
				// Note that for both the channel and calcium maps, we
				// do the actual conversion through an array of interface
				// objects which handle scaling and offsets. The
				// scaling for the channelMap is from gmax and CoInit, 
				// and the baseline for calciumMap is CaBasal and CoInit,
				// and it uses calciumScale for scaling.
};

#endif // _KINETIC_MANAGER_H
