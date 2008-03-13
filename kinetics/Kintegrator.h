/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _KINTEGRATOR_H
#define _KINTEGRATOR_H
class Kintegrator
{
	public:
		Kintegrator();

		static bool getIsInitialized( Eref e );
		static string getMethod( Eref e );
		static void setMethod( const Conn* c, string method );
		void innerSetMethod( const string& method );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		static void allocateFunc( const Conn* c, vector< double >* y );
		void allocateFuncLocal( vector< double >*  y );
		static void processFunc( const Conn* c, ProcInfo info );
		void innerProcessFunc( Eref e, ProcInfo info );
		static void reinitFunc( const Conn* c, ProcInfo info  );

	private:
		bool isInitialized_;
		string method_;
		vector< double >* y_;
		vector< double > yprime_;
};
#endif // _KINTEGRATOR_H
