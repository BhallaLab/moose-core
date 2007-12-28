#ifndef _Table_h
#define _Table_h

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

enum TableModes { TAB_IO, TAB_LOOP, TAB_ONCE, TAB_BUF, TAB_SPIKE, TAB_FIELDS, TAB_DELAY };

class Table: public Interpol
{
	public:
		Table();

		////////////////////////////////////////////////////////////
		// Here are the interface functions for the MOOSE class
		////////////////////////////////////////////////////////////
		static void setInput( const Conn& c, double value );
		static double getInput( const Element* e );
		static void setOutput( const Conn& c, double value );
		static double getOutput( const Element* e );
		static void setMode( const Conn& c, int value );
		static int getMode( const Element* e );
		static void setStepsize( const Conn& c, double value );
		static double getStepsize( const Element* e );
		static void setStepMode( const Conn& c, int value );
		static int getStepMode( const Element* e );
		static double getLookup( const Element* e, const double& x );

		////////////////////////////////////////////////////////////
		// Here are the Table Destination functions
		////////////////////////////////////////////////////////////
		static void sum( const Conn& c, double val );
		static void prd( const Conn& c, double val );
		static void input2( const Conn& c, double y, unsigned int x );
		static void process( const Conn& c, ProcInfo p );
		static void reinit( const Conn& c, ProcInfo p );
		static void tabop( const Conn& c, char op, double min, double max );

		////////////////////////////////////////////////////////////
		// Here are the internal functions
		////////////////////////////////////////////////////////////
		void innerProcess( Element* e, ProcInfo p );
		void innerReinit( const Conn& c, ProcInfo p );
		unsigned long expandTable( Element* e, double size );
		void innerTabop( char op, double min, double max );
		void doOp( char op, unsigned int istart, unsigned int istop );

	private:
		double input_;
		double output_;
		double stepSize_;
		int stepMode_;
		double sy_;
		double py_;
		double lastSpike_;
		unsigned int counter_;
};

extern const Cinfo* initTableCinfo();

#endif // _Table_h
