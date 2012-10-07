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
		void setInputOffset( double offset );
		double getInputOffset() const;
		void setOutputOffset( double offset );
		double getOutputOffset() const;
		void setScale( double scale );
		double getScale() const;
		double getOutput() const;

		////////////////////////////////////////////////////////////
		// Here are the Destination functions
		////////////////////////////////////////////////////////////
		void input( double val );
		void innerProcess();
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		/*
		static void setup( const Eref& e, const Qinfo* q, 
			string molName, double scale, 
			double inputOffset, double outputOffset );
		static void build( const Eref& e, const Qinfo* q);
		*/

		////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	private:
		double output_;
		double inputOffset_;
		double outputOffset_;
		double scale_;
		string molName_; /// Used for placeholding in cellreader mode.
		double sum_;
		unsigned int counter_;
};

#endif // _Adaptor_h
