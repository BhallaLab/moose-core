/*******************************************************************
 * File:            PropertyReader.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-08 09:25:53
 ********************************************************************/

#ifndef _PROPERTYREADER_CPP
#define _PROPERTYREADER_CPP
#include "PropertyReader.h"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "StringUtil.h"


map<std::string, PropertyReader*>& PropertyReader::getReaders()
{
    static map<std::string, PropertyReader* > readers_;
    return readers_;    
}

/// Constructs a property map by reading the contents of the file fileName
PropertyReader::PropertyReader(std::string fileName)
{
    ifstream input(fileName.c_str());
    char buffer[MAXCHAR];
    int lineNo = 0;
    fileName_ = "";
    if (!input)
    {
        cerr << "ERROR: failed to open property file: " << fileName << endl;
        return;
    }
    
    while(input.good())
    {
        input.getline(buffer, MAXCHAR);
        std::string line(buffer);
        ++lineNo;
        if ((trim(line)).length() == 0 )
        {
            continue;
        }
        
        size_t delimPos = line.find('=');
        if (delimPos == std::string::npos)
        {
            cerr << "ERROR: PropertyReader::PropertyReader( "<< fileName <<" ) :: no separator (=) in line " << lineNo << endl;
            return;
        }
        
        std::string key=line.substr(0,delimPos);
        std::string value =line.substr(delimPos+1);        
        key = trim(key);
        value = trim(value);
        properties_[key] = value;        
    }
    fileName_ = fileName;    
}

PropertyReader::~PropertyReader()
{
    cout << "Deleting PropertyReader object for file " << fileName_ << endl;
}

/// Returns the property value fore key name, if no such property exists, an empty std::string is returned.
const std::string PropertyReader::getProperty(std::string name) const
{
    map < std::string, std::string > :: const_iterator i;

    i = properties_.find(name);
    if ( i == properties_.end() )
    {
        cerr << "ERROR: PropertyReader::getProperty( " << name << " ) :: No such key exists in property map." << endl;
        return "";        
    }
    else 
    {
        return i->second;//i->second;
    }
}

/// Returns a list of the current property keys
vector <std::string>* PropertyReader::getPropertyNames()
{
    vector <std::string>* propNames = new vector<std::string>();
    for ( map<std::string,std::string>::iterator i = properties_.begin(); i != properties_.end(); ++i )
    {
        propNames->push_back(i->second);
    }
    return propNames;    
}

///Returns the file name associated with this property map
const std::string PropertyReader::getFileName()
{
    return fileName_;
}

/// Returns a PropertyReader object reference corresponding to file fileName
const PropertyReader& PropertyReader::getPropertyReader(std::string fileName)
{
    map <std::string, PropertyReader*>::iterator i = getReaders().find(fileName);
    
    if ( i == getReaders().end())
    {
        getReaders()[fileName] = new PropertyReader(fileName);
    }
    PropertyReader* reader = getReaders()[fileName];
    
    return *reader;    
}

/// Return value of property key as specified in property file fileName
const std::string PropertyReader::getProperty(std::string fileName, std::string key) 
{
    return getPropertyReader(fileName).getProperty(key);
}

void PropertyReader::clearAll()
{
    for ( map < std::string, PropertyReader* >::iterator i = getReaders().begin(); i != getReaders().end(); ++i )
    {
        delete i->second;
    }
    getReaders().clear();    
}

// int main(void)
// {
//     cout << PropertyReader::getProperty("moose.cfg","AutoSchedule") << endl;    
//     PropertyReader::clearAll();
//     return 0;    
// }

#endif
