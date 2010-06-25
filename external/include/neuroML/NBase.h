#ifndef _NBASE_H
#define _NBASE_H
#include <string>
#include <libxml/xmlreader.h>
#include "NCell.h"
using namespace std;
class NCell;
class NBase
{
	public:
		NCell* readNeuroML(string filename);
		static void setPaths( vector< string > paths );
		static const vector< string >& getPaths();
		
	protected:
		//std::string id;
		//std::string name;
		//int level;
		//int version;
		NCell cell_;
		//bool setnamespaces;
		static vector< string > paths_;

};
#endif // _NBASE_H
