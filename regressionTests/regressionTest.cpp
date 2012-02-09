/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include <unistd.h> // need Windows-specific stuff too.


void rtTable();
void rtFindModelType();
void rtReadKkit();
void rtRunKkit();
void rtReadCspace();
void rtRunCspace();
void rtReacDiff();
void rtReadKkitModels( const string& modelname, const char** path,
	const char** field, const double* value, unsigned int num );
void rtHHnetwork( unsigned int numCopies );

extern void testGsolver( string modelName, string plotName,
	double plotDt, double simtime );

void regressionTests()
{
	//char* cwd = get_current_dir_name();
	// get_current_dir_name is not available on all platforms
	char *cwd = getcwd(NULL,0); // same behaviour but this also resolves symbolic links
	string currdir = cwd;
	free( cwd );
	if ( currdir.substr( currdir.find_last_of( "/" ) ) != 
		"/regressionTests" ) {
		cout << "\nNot in regression test directory, so skipping them\n";
		return;
	}
	cout << "\nRegression Tests:";
	rtTable();
	rtFindModelType();
	rtReadKkit();

	static const char* acc8path[] = { 
		"/kkit/kinetics/Ca", 
		"/kkit/kinetics/PLA2/PIP2-PLA2*/kenz"
	};
	static const char* field[] = { "concInit", "Km" };
	static const double value[] = { 0.08e-3, 20e-3 };
	rtReadKkitModels( "acc8.g", acc8path, field, value, 2 );
	rtReadKkitModels( "Anno_acc8.g", acc8path, field, value, 2 );
	rtRunKkit();
	rtReadCspace();
	rtRunCspace();
	rtReacDiff();
	rtHHnetwork( 10 );

	testGsolver( "reac", "A.Co", 0.1, 100 );
	cout << endl;
}
