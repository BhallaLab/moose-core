/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Average
{
	public:
		Average();

		static void process( const Conn& c, ProcInfo p );
		static void input( const Conn& c, double v );
		void inputLocal( double v );
		double mean( ) const;

		static double getTotal( const Element* e );
		static void setTotal( const Conn& c, double v );
		static double getBaseline( const Element* e );
		static void setBaseline( const Conn& c, double v );
		static unsigned int getN( const Element* e );
		static void setN( const Conn& c, unsigned int v );
		static double getMean( const Element* e );
		static void processFunc( const Conn& c, ProcInfo info );
		void processFuncLocal( Element* e, ProcInfo info );
		static void reinitFunc( const Conn& c, ProcInfo info );
		void reinitFuncLocal( );
	private:
		double total_;
		double baseline_;
		unsigned int n_;
};
