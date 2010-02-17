#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include <libxml/xmlreader.h>
#include <fstream>
#include "NBase.h"
#include "NCell.h"
using namespace std;
vector< string > NBase::paths_;

/* this is supposed to be the first function to invoke. it takes filename and returns NCell class */
NCell* NBase::readNeuroML(string filename)
{	
	ifstream fin;
	string path;
	unsigned int i;
	const vector< string >& paths = NBase::getPaths();
	for ( i = 0; i < paths.size(); i++ ) {
		path = paths[ i ] + "/" + filename;
		fin.open( path.c_str() );
		if ( fin ){
			fin.close();
			cout << "The file " << filename << " is loaded from "<< path << endl;
			break;
		}
	}
	if( i == paths.size() )
	{
		fin.clear();
		cerr<< "The file " << filename << " is not found in any of the paths" << endl ;
		return 0;
	}
	xmlDocPtr xmlDoc = xmlParseFile(path.c_str());
	xmlXPathContextPtr context = xmlXPathNewContext(xmlDoc);
  	xmlTextReaderPtr readerPtr = xmlReaderForFile(path.c_str(), NULL, 0);
	/*if (register_namespaces() < 0){	
    	    	cerr << "Error: unable to register Namespaces" << endl;
     		xmlXPathFreeContext(context); 
     		xmlFreeDoc(xmlDoc); 
  	}*/
	cell_.setXmldoc(xmlDoc);
	cell_.setContext(context);
	cell_.setReaderptr(readerPtr);
	return &cell_;
}

void NBase::setPaths( vector< string > paths )
{ paths_ = paths; }
const vector< string >& NBase::getPaths()
{ return paths_; }

