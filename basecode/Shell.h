/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SHELL_H
#define _SHELL_H

class Shell: public Data
{
	public:
		Shell();
		void process( const ProcInfo* p, const Eref& e );
		void setName( string name );
		string getName() const;
		void setQuit( bool val );
		bool getQuit() const;

		void start( double runTime );

		void handleGet( Eref e, const Qinfo* q, const char* arg );
		const char* getBuf() const;
		static const char* buf();
		static const ProcInfo* procInfo();
		/**
		 * Assigns the hardware availability. Assumes that each node will
		 * have the same number of cores available.
		 */
		void setHardware( bool isSingleThreaded, 
			unsigned int numCores, unsigned int numNodes );

		/**
		 * Stub for eventual function to handle load balancing. This must
		 * be called to set up default groups.
		 */
		void loadBalance();
		unsigned int numCores();

		// Sets up clock ticks. Essentially is a call into the 
		// Clock::setupTick function, but may be needed to be called from
		// the parser so it is a Shell function too.
		void setclock( unsigned int tickNum, double dt, unsigned int stage );

		static const Cinfo* initCinfo();
	private:
		string name_;
		vector< char > getBuf_;
		bool quit_;
		bool isSingleThreaded_;
		unsigned int numCores_;
		unsigned int numNodes_;
		static ProcInfo p_; 
			// Shell owns its own ProcInfo, has global thread/node info.
			// Used to talk to parser and for thread specification in
			// setup operations.
};

extern bool set( Eref& dest, const string& destField, const string& val );

extern bool get( const Eref& dest, const string& destField );

#endif // _SHELL_H
