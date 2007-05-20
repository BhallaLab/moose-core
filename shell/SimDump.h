/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SimDump_h
#define _SimDump_h

#include <string>
#include <vector>

class Element; 	// Forward declaration

using namespace std;

unsigned int parseArgs( const string& in, vector< string >& out );

class SimDumpInfo {
	public:
		SimDumpInfo( const string& oldObject, const string& newObject,
			const string& oldFields, const string& newFields);

		// Takes info from simobjdump
		void setFieldSequence( int argc, const char** argv );

		// Sets the fields from the simundump arg list.
		bool setFields( Element* e, int argc, const char** argv);

		string oldObject() {
			return oldObject_;
		}

		string newObject() {
			return newObject_;
		}
		

	private:
		string oldObject_;
		string newObject_;
		map< string, string > fields_;
		vector< string >fieldSequence_;
};



class SimDump
{
	public:
		SimDump();
		void simObjDump( int argc, const char** argv );
		void simUndump( int argc, const char** argv );
	private:
		map< string, SimDumpInfo* > dumpConverter_;
};
#endif // _SimDump_h
