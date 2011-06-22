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
void rtHHnetwork( unsigned int numCopies );

void regressionTests()
{
	char* cwd = get_current_dir_name();
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
//	rtReadKkit(); // Commented out because of ongoing coding.
	rtHHnetwork( 10 );
	cout << endl;
}
