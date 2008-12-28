#ifndef _Adaptor_h
#define _Adaptor_h

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Adaptor
{
	public:
		Adaptor();

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setInputOffset( const Conn* c, double value );
		static double getInputOffset( Eref e );
		static void setOutputOffset( const Conn* c, double value );
		static double getOutputOffset( Eref e );
		static void setScale( const Conn* c, double value );
		static double getScale( Eref e );
		static double getOutput( Eref e );

		////////////////////////////////////////////////////////////
		// Here are the Adaptor Destination functions
		////////////////////////////////////////////////////////////
		static void input( const Conn* c, double val );
		static void process( const Conn* c, ProcInfo p );
		static void reinit( const Conn* c, ProcInfo p );
		static void setup( const Conn* c,
			string molName, double scale, 
			double inputOffset, double outputOffset );
		static void build( const Conn* c );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		void innerProcess( Eref e, ProcInfo p );
		void innerReinit( const Conn* c, ProcInfo p );

	protected:
		double output_;
		double inputOffset_;
		double outputOffset_;
		double scale_;
		string molName_; /// Used for placeholding in cellreader mode.
		double sum_;
		unsigned int counter_;
};

extern const Cinfo* initAdaptorCinfo();

#endif // _Adaptor_h
