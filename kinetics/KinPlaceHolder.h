#ifndef _KIN_PLACE_HOLDER_H
#define _KIN_PLACE_HOLDER_H

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class is a placeholder to handle the setup of kinetic models 
 * through readcell or equivalent, where the setup happens on a per-
 * cell-compartment basis similar to a channel. It is necessary to 
 * do this through a placeholder since the SigNeur needs a birds-eye
 * view of the whole cell in order to work out how to decompose the
 * model.
 */

class KinPlaceHolder
{
	public:
		KinPlaceHolder();

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static Id getProto( Eref e );
		static double getLambda( Eref e );
		static string getMethod( Eref e );
		static double getLoadEstimate( Eref e );
		static unsigned int getMemEstimate( Eref e );
		static double getSigComptLength( Eref e );
		static unsigned int getNumSigCompts( Eref e );

		////////////////////////////////////////////////////////////
		// Here are the Adaptor Destination functions
		////////////////////////////////////////////////////////////

		/// Make the model.
		static void build( const Conn* c );

		/**
		 * Set up info needed for KinPlaceHolder. We need all this
		 * info to do so. Also implicit info is parent compt, 
		 * hence dia and length. Rather than require a specific
		 * sequence of field assignments, less surprise with the
		 * single setup call.
		 */
		static void setup( const Conn* c, 
			Id proto, double lambda, string method );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		void innerSetup( Eref e,
			Id proto, double lambda, string method );
		void innerBuild( Eref e );

		void assignDiffusion( Element* e );
		void connectAdaptors( Element* e );
		void setupSolver( Element* e );
		void connectFluxes( Element* e );
		void assignVolumes( Element* dup ) const;

	protected:
		/// The prototype object, typically in /library or /proto
		Id proto_;

		/// Length constant of kin model, loaded in from readcell
		double lambda_;

		/// Numerical method to use for kin model.
		string method_;

		/**
		 * Estimate of computational load for kin model.
		 * Expressed roughly as # of flops per second sim time
		 */
		double loadEstimate_;

		/// Estimate of number of bytes needed to define kin model.
		unsigned int memEstimate_;
			
		/// Length of signaling compartment, after fitting to el compt.
		double sigComptLength_;

		/// Volume of signaling compartment, after fitting to el compt.
		double sigComptVolume_;

		/// Number of signaling compts, from el compt length and lambda
		unsigned int numSigCompts_;
};

extern const Cinfo* initKinPlaceHolderCinfo();

#endif // _KIN_PLACEHOLDER_H
