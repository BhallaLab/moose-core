/*******************************************************************
 * File:            Property.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-04 16:55:24
 ********************************************************************/
/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment,
 ** also known as GENESIS 3 base code.
 **           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU General Public License version 2
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#ifndef _PROPERTY_CPP
#define _PROPERTY_CPP

#include "utility.h"
#include "Property.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

const char* const Property::SIMPATH = "SIMPATH";
const char* const Property::SIMNOTES = "SIMNOTES";
const char* const Property::DOCPATH = "DOCPATH";
const char* const Property::AUTOSCHEDULE = "AUTOSCHEDULE";
const char* const Property::CREATESOLVER = "CREATESOLVER";
const char* const Property::HOME = "HOME";

const int Property::XML_FORMAT = 1;
const int Property::PROP_FORMAT = 0;
map <string, string>& Property::properties()
{
    static map <string, string>* properties_ = new map<string, string>();
    return *properties_;
}

PathUtility* Property::simpathHandler_ = new PathUtility(".");

bool Property::initialized_ = false;

void Property::initDefaults()
{
    if (!simpathHandler_){
        simpathHandler_ = new PathUtility(".");
    }
    properties()[AUTOSCHEDULE] = "true";
    properties()[CREATESOLVER] = "true";
    properties()[SIMPATH] = ".";
    properties()[SIMNOTES] = "notes";
    properties()[DOCPATH] = "doc";
    properties()[HOME] = "~";    
    char * home = getenv(HOME);
    if (( home != NULL ) && (home[0] != '\0'))
    {
        simpathHandler_->addPath(string(home));
        properties()[SIMPATH] = simpathHandler_->getAllPaths();
    }
}

string Property::getProperty(string key)
{
    return properties()[key];
}

void Property::setProperty(string key, string value)
{
    properties()[key] = value;
}

void Property::addSimPath(string path)
{
    if (!initialized_){
        initialize("", 0); // no need to read file
    } else {
        simpathHandler_->addPath(path);
    }
    cout << simpathHandler_->getAllPaths() << endl;
    properties()[SIMPATH] = simpathHandler_->getAllPaths();
}

const string Property::getSimPath()
{
    return properties()[SIMPATH];
}

void Property::setSimPath(string paths)
{
    cout << "changing simpath:" << paths << endl;
    if (simpathHandler_){
        delete simpathHandler_;
    }
    simpathHandler_ = new PathUtility(paths);
    properties()[SIMPATH] = simpathHandler_->getAllPaths();
    cout << "changing simpath:" << paths << endl;

}
/**
   Reads the property values from environment.  This function should
   be called just after initDefaults if required.  The overriding
   order should be -

   defaults < environment < property file < commandline.

   Note that in command line we can specify a propety file that has
   content conflicting with the other properties specified in command
   line. In this case care should be taken to make sure the command
   line arguments take precedence.
*/
void Property::readEnvironment()
{
    map <string, string>::iterator i;
    
    for ( i = properties().begin(); i != properties().end(); ++i )
    {
        string key = i->first;
        char * env;
        
        // Special treatment for SIMPATH - it should be extended
        // rather than replaced
        if ( key == SIMPATH )
        {
            
            env = getenv(SIMPATH);            
            if (( env != NULL ) && (env[0] != '\0'))
            {
                simpathHandler_->addPath(string(env));
            }            
            continue;
        }
        
        env = getenv(key.c_str());
        if (( env != NULL ) && ( env[0] != '\0') ){
            // if environment variable exists, then it should override the defaults
            string value(env);            
            properties()[key] = env;
        }
    }    
}
/**
   Set up the properties in this sequence:
    - set the default values
    - override defaults with anything specified in environment
    - override with anything available in property file only if a proper file is specified
*/
void Property::initialize(string fileName, int format)
{
    if (initialized_){
        return;
    }
    Property::initDefaults();
    Property::readEnvironment();
    if (fileName.length() > 0)
    {
        readProperties(fileName, format);
    }
    initialized_ = true;
}

/**
   Read properties from file, which can be in either XML or Key = Value format.
   @param format is Property::XML_FORMAT to read xml configuration
   Property::PROP_FORMAT to read key = value format.   
*/
   
int Property::readProperties(string fileName, int format)
{
    int returnValue = 1;
    string simpath = properties()[SIMPATH];
    
    if ( format == XML_FORMAT)
    {
        returnValue = readXml(fileName);
    }
    else if (format == PROP_FORMAT )
    {
        returnValue = readProp(fileName);
    }
    // append simpath from file to the existing simpath
    if ((simpath != properties()[SIMPATH] )
        && ( properties()[SIMPATH].length() > 0 ))
    {
        properties()[SIMPATH] = simpath+ PathUtility::PATH_SEPARATOR + properties()[SIMPATH];
    }
    
    return returnValue;
}

/**
   Read a property file in XML format - this is only for future
   consideration.  Introduces dependency on libxml++
*/
int Property::readXml(string fileName)
{
     // Check if the input configuration file exists
    fstream fin;
    fin.open(fileName.c_str(),ios::in);
    if( !fin.is_open() )
    {
        return 1;
    }
    fin.close();
        
    // Use xml configuration file only if libxml is there
#ifdef HAS_LIBXML
#include <libxml++/libxml++.h>
    try
    {
//         Sample file:
//             <?xml version="1.0" encoding="UTF-8"?>
//             <MooseConfig>
//                      <AutoSchedule>true</AutoSchedule>
//                      <CreateSolvers>false</CreateSolvers>
//             </MooseConfig>
//
// Further depth is not allowed now in order to keep the configuration simplest
        
        
        xmlpp::DomParser parser;
        parser.set_substitute_entities(); //We just want the text to be resolved/unescaped automatically.
        parser.parse_file(fileName);
        if(parser)
        {
            const xmlpp::Node* pNode = parser.get_document()->get_root_node(); //deleted by DomParser.
            xmlpp::Node::NodeList propertyNodes = pNode->get_children();
            // process only the elements at first level and the text associated with them.
            for(xmlpp::Node::NodeList::iterator iter = propertyNodes.begin(); iter != propertyNodes.end(); ++iter)
            {
                xmlpp::Element* element = dynamic_cast<xmlpp::Element*>( *iter );
                if ( element != NULL )
                {
                    xmlpp::TextNode * textNode = element->get_child_text();
                    if ( textNode && !textNode->is_white_space() )
                    {                        
                        std::string value =  trim((std::string)textNode->get_content());
                        if (value.length() > 0 )
                        {
                            std::string key = element->get_name();
                            properties[key] = value;
                        }   
                    }                    
                }                
            }            
        }
        return 1;        
    }
    catch(const std::exception& e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return 2;        
    }
#endif // HAS_LIBXML
    return 2;    
}


/**
   Reads a property file in the popular key=value\n format
*/
int Property::readProp(string fileName)
{
    ifstream input(fileName.c_str());
    char buffer[BUF_SIZE];
    int lineNo = 0;
    if (!input)
    {
        initialized_ = true;
        return 1;
    }
    
    while(input.good())
    {
        input.getline(buffer, BUF_SIZE);
        std::string line(buffer);
        ++lineNo;
        if ((trim(line)).length() == 0 )
        {
            continue;
        }
        
        size_t delimPos = line.find('=');
        if (delimPos == std::string::npos)
        {
            cout << "ERROR: PropertyReader::parsePropFile( "<< fileName <<" ) :: no separator (=) in line " << lineNo << endl;
            return 2;
        }
        
        std::string key=line.substr(0,delimPos);
        key = trim(key);
        string value = "";
        
        std::string value_part = trim(line.substr(delimPos+1));
        while ( value_part.length() > 0 )
        {
            if ( value_part[value_part.length()-1] != '\\')
            {
                value += value_part;                
                break;
            }
            else
            {
                value += value_part.substr(0,value_part.length()-1);
                value_part = trim(line.substr(delimPos+1));
                input.getline(buffer, BUF_SIZE);
                string my_line( buffer );
                ++lineNo;
                value_part = trim(my_line);                
            }
        }
        
        properties()[key] = value;        
    }
    return 0;    
}

/**
   returns a vector containing all the unique property keys
*/

vector <string>& Property::getKeys()
{
    static vector<string> result;
    
    map <string, string> :: iterator i;
    for ( i = properties().begin(); i != properties().end(); ++i )
    {
        bool found = false;
        
        for ( unsigned j = 0; j < result.size(); j++ )
        {
            if ( result[j] == (i->first)) 
            {
                found = true;                
                break;
            }
        }
        if (!found)
            result.push_back(i->first);
    }
    return result;
}

#ifdef TEST_PROPERTY
int main(int argc, char **argv)
{
    
    Property::initDefaults();
    vector <string> keys = Property::getKeys();
    cout << "Testing class Property:-\n Initial defaults -" << endl;
    
    for ( unsigned i = 0; i < keys.size(); ++i )
    {
        cout << "Property::" << keys[i] << " = " << Property::getProperty(keys[i]) << endl;
    }
    
    Property::readEnvironment();
    cout << "After updating from ENV -" << endl;
    
    for ( unsigned i = 0; i < keys.size(); ++i )
    {
        cout << "Property::" << keys[i] << " = " << Property::getProperty(keys[i]) << endl;
    }

    Property::initialize("test.prop", Property::PROP_FORMAT);
    cout << "After updating from file: test.prop -" << endl;

    
    for ( unsigned i = 0; i < keys.size(); ++i )
    {
        cout << "Property::" << keys[i] << " = " << Property::getProperty(keys[i]) << endl;
    }
    return 0;
}

#endif // TEST_PROPERTY

#endif // _PROPERTY_CPP
