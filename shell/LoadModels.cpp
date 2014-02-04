/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <fstream>
#include "header.h"
#include "Shell.h"
#include "../utility/strutil.h"
#include "LoadModels.h" // For the ModelType enum.

#include "../biophysics/ReadCell.h"
#include "../kinetics/ReadKkit.h"
#include "../kinetics/ReadCspace.h"

ModelType findModelType( string filename, ifstream& fin, string& line )
{
	if ( filename.substr( filename.length() - 2 ) == ".p" )
		return DOTP;

	getline( fin, line );
        line = trim(line);
	if ( line == "//genesis" ) {
		getline( fin, line );
                line = trim(line);
		if ( line.substr( 0, 7 ) == "// kkit" )
			return KKIT;
	}
	if ( line.substr( 0, 9 ) == "//  DOQCS" ) {
		while ( getline( fin, line ) )
		{
                        line = trim(line);
			if ( line.substr( 0, 7 ) == "// kkit" )
				return KKIT;
		}
	}
	
	unsigned long pos = line.find_first_of( ":" );
	string copyLine= line;
	if (pos != string::npos ){
		copyLine = line.substr(pos+2);
	}

	if ( copyLine.length() >= 6 && copyLine[0] == '|' && copyLine[5] == '|' )
		return CSPACE;

	return UNKNOWN;
}

/**
 * Identify parent Id from path that can optionally have the
 * model name as the last part. Pass back the parent Id,
 * and the model name.
 * Returns true on success.
 * Cases:
 *	First set: we want to fill in "model" as modelName
 *		""		where we want <cwe>/model
 *		"/"		where we want /model.
 *		"/foo"	where foo exists, so we want /foo/model
 *		"foo"	where <cwe>/foo exists, and we want <cwe>/foo/model
 *	Second set: the last part of the path has the model name.
 *		"bar"	where we want	<cwe>/bar
 *		"/bar"	where we want /bar
 *		"/foo/bar"	where we want /foo/bar
 *		"foo/bar"	where we want <cwe>/foo/bar
 */
bool findModelParent( Id cwe, const string& path,
	Id& parentId, string& modelName )
{
	modelName = "model";
	string fullPath = path;

	if ( path.length() == 0 ) {
		parentId = cwe;
		return 1;
	}

	if ( path == "/" ) {
		parentId = Id();
		return 1;
	}

	if ( path[0] != '/' ) {
		string temp = cwe.path(); 
		if ( temp[temp.length() - 1] == '/' )
			fullPath = temp + path;
		else
			fullPath = temp + "/" + path; 
	}

	Id paId( fullPath );
	if ( paId == Id() ) { // Path includes new model name
		string::size_type pos = fullPath.find_last_of( "/" );
		assert( pos != string::npos );
		string head = fullPath.substr( 0, pos );
		Id ret( head );
		// When head = "" it means paId should be root.
		if ( ret == Id() && head != "" && head != "/root" )
			return 0;
		parentId = ret;
		modelName = fullPath.substr( pos + 1 );
		return 1;
	} else { // Path is an existing element.
		parentId = paId;
		return 1;
	}
	return 0;
}

/// Returns the Id of the loaded model.
Id Shell::doLoadModel( const string& fileName, const string& modelPath, const string& solverClass )
{
	ifstream fin( fileName.c_str() );
	if ( !fin ) {
		cerr << "Shell::doLoadModel: could not open file " << fileName << endl;
		return Id();
	}

	string modelName;
	Id parentId;

	if ( !( findModelParent ( cwe_, modelPath, parentId, modelName ) ) )
		return Id();

	string line;	
	switch ( findModelType( fileName, fin, line ) ) {
		case DOTP:
			{
				ReadCell rc;
				return rc.read( fileName, modelName, parentId );
				return Id();
			}
		case KKIT: 
			{
				string sc = solverClass;
				ReadKkit rk;
				Id ret = rk.read( fileName, modelName, parentId, sc);
				return ret;
			}
			break;
		case CSPACE:
			{
				string sc = solverClass;
				ReadCspace rc;
				Id ret = rc.readModelString( line, modelName, parentId, sc);
				rc.makePlots( 1.0 );
				return ret;
			}
		case UNKNOWN:
		default:
			cout << "Error: Shell::doLoadModel: File type of '" <<
				fileName << "' is unknown\n";
	}
	return Id();
}
