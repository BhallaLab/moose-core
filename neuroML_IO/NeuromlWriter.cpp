/*******************************************************************
 * File:            NeuromlWriter.cpp
 * Description:      
 * Author:          Siji P George
 * E-mail:          siji.suresh@gmail.com
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "moose.h"
#include <math.h>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "element/Neutral.h"
#include "biophysics/Compartment.h"
#include "biophysics/HHChannel.h"
#include "NeuromlWriter.h"

/**
*  write a Model after validation
*/

void NeuromlWriter::writeModel( string filepath,Id location )
{

	/*string::size_type loc;
	while ( ( loc = filepath.find( "\\" ) ) != string::npos ) {
		filepath.replace( loc, 1, "/" );
	}
	/* allows to write filename with extensions xml,zip,bz2 and gz. if no 
	extension is given then .xml is the default one. */
	/*string fName = filepath;
	if ( filepath[0]== '~' ){
		cerr << "Error : Replace ~ with absolute path " << endl;
		return ;
	}
	//string::size_type tilda_pos = fName.
	string::size_type slash_pos = fName.find_last_of("/");
	fName.erase( 0,slash_pos + 1 );  
	//cout<<"filename:"<<filename<<endl;
	
	vector< string > extensions;
	extensions.push_back( ".xml" );
	extensions.push_back( ".zip" );
	extensions.push_back( ".bz2" );
	extensions.push_back( ".gz" );
	vector< string >::iterator i;
	for( i = extensions.begin(); i != extensions.end(); i++ ) {
		string::size_type loc = fName.find( *i );
		if ( loc != string::npos ) {
		     	int strlen = fName.length(); 
			fName.erase( loc,strlen-loc );
			break;
		}
	}
	if ( i == extensions.end() && fName.find( "." ) != string::npos ) {
		string::size_type loc;
		while ( ( loc = fName.find( "." ) ) != string::npos ) {
			fName.replace( loc, 1, "_" );
		}
	}
	if ( i == extensions.end() )
		filepath += ".xml";
	//SBMLDocument* sbmlDoc = 0;
  	bool fileok = false;
	//sbmlDoc = createModel( fName ); 
  	//fileok  = validateModel( sbmlDoc );
	if ( fileok ) 
		writeModel( sbmlDoc, filepath );
    	delete sbmlDoc;
	if ( !fileok ) {
		cerr << "Errors encountered " << endl;
		return ;
	}*/
	
}
