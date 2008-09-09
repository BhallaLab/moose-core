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


/**
 * enum of the different compartment types, worked out by analyzing the
 * cell model. Each of these will get assigned a distinct signaling model
 * and possibly different numberical method.
 */
enum CompartmentCategory { SOMA, DEND, SPINE, SPINE_NECK };

/**
 * Data structure that organizes the composite model.
 * Assumes arrays for each of the signaling models.
 */
class TreeNode {
	public:
		TreeNode( Id c, unsigned int p, CompartmentCategory cc )
			:
				compt( c ),
				parent( p ),
				sigStart( 0 ),
				sigEnd( 0 ),
				category( cc )
		{;}

		Id compt; /// Id of electrical compartment on this node.

		/// Parent compartment treeNode index. Empty for soma.
		unsigned int parent; 

		Id sigModel; /// Parent of array of sigModels.
		unsigned int sigStart; /// Indices of sigModel entries.
		unsigned int sigEnd; /// Indices of sigModel entries.

		/// type of compartment and hence sigModel to use.
		CompartmentCategory category; 
};

/**
 * SigNeur: A multiscale model builder, aka a robot.
 * This potentially interfaces 4 models: An electrical cell model, and
 * chemical models for spines, dendrites and soma. The latter may be
 * 3-D stochastic, ODE, and genetic respectively, but for now the
 * class only handles ODE versions.
 *
 * Constructs a copy of the cell model on itself. The cell has its own
 * cell manager. Then there is an array of dends, of spines and 
 * soma models. Each of these incorporates diffusion, which may be
 * between categories. The whole mess sits on a kinetic manager as it
 * will start out by solving it using rk5. A 300-compartment model will
 * have ~50,000 molecular species. This is about 10 times the biggest I
 * have tried so far.
 * There is also an array of adaptors, one per cell model compt.
 * In due course need to put a distinct kind of solver for the spines,
 * either 3D monte carlo or Gillespie. Fortunately these will be separated
 * and hence can run on distinct processors.
 */
class SigNeur
{
	public:
		SigNeur();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		
		// Object selection
		static void setCellProto( const Conn* c, Id value );
		static Id getCellProto( Eref e );
		static void setSpineProto( const Conn* c, Id value );
		static Id getSpineProto( Eref e );
		static void setDendProto( const Conn* c, Id value );
		static Id getDendProto( Eref e );
		static void setSomaProto( const Conn* c, Id value );
		static Id getSomaProto( Eref e );

		// These are readonly.
		static Id getCell( Eref e );
		static Id getSpine( Eref e );
		static Id getDend( Eref e );
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

		/// Characteristic length for linear diffusion.
		static void setLambda( const Conn* c, double value );
		static double getLambda( Eref e );

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
		bool traverseCell( Eref me );
		Id findSoma( const vector< Id >& compts );
		void buildTree( Id soma, const vector< Id >& compts );
		// void innerBuildTree( unsigned int parent, Eref e, int msg );
		void innerBuildTree( unsigned int parent, Eref paE, Eref e, 
			int msg1, int msg2 );
		void assignSignalingCompts();
		void makeSignalingModel();
		static CompartmentCategory guessCompartmentCategory( Eref e );

	private:
		Id cellProto_; /// Prototype cell electrical model
		Id spineProto_; /// Prototype spine signaling model
		Id dendProto_; /// prototype dendrite signaling model
		Id somaProto_; /// prototype soma signaling model
		Id cell_; /// cell electrical model base.
		Id spine_; /// spine signaling model array base
		Id dend_; /// dendrite signaling model array base
		Id soma_; /// soma signaling model array base
		string cellMethod_;
		string spineMethod_;
		string dendMethod_;
		string somaMethod_;
		double Dscale_;	/// Diffusion scale factor.
		double lambda_; /// Length constant for diffusion.
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
		vector< TreeNode > tree_;
};

#endif // _KINETIC_MANAGER_H
