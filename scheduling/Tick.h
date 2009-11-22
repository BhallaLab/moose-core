/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Tick_h
#define _Tick_h
// Should be derived from projection, since we need to maintain a 
// text wildcard list of targets, and possibly manipulate it.
class Tick
{
	public:
		Tick();
		virtual ~Tick();

		bool operator<( const Tick& other ) const;
		bool operator==( const Tick& other ) const;

		///////////////////////////////////////////////////////
		// Functions for handling field assignments.
		///////////////////////////////////////////////////////
		void setDt( double v );
		double getDt() const;
		void setStage( unsigned int v );
		unsigned int getStage() const;
		double getNextTime() const;
		void setPath( string v );
		string getPath() const;

		///////////////////////////////////////////////////////
		// Functions for handling messages
		///////////////////////////////////////////////////////

		void increment( Eref e, ProcInfo* p ) const;

		/**
		 * Reinit is used to set the simulation time back to zero for
		 * itself, and to trigger reinit in all targets, and to go on
		 * to the next tick
		 */
		void reinit( Eref e ) const;

		///////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
		bool running_;
		int callback_;
		double dt_;
		unsigned int stage_;
		double nextTime_;
		double nextTickTime_;
		bool next_; /// Flag to show if next_ tick is present
		bool terminate_;
		string path_;/// \todo Perhaps we delete this field
};

#endif // _Tick_h
