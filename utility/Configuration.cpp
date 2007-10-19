/*******************************************************************
 * File:            Configuration.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 08:28:39
 ********************************************************************/
#ifndef _CONFIGURATION_CPP
#define _CONFIGURATION_CPP
#include <iostream>
#include <fstream>
#include "StringUtil.h"

#include "Configuration.h"
#include "PropertyReader.h"
/** The allowed property names */
const std::string Configuration::AUTOSCHEDULE = "AutoSchedule";
const std::string Configuration::CREATESOLVER = "CreateSolver";

/**
   Reads the conFile and stores the property values.
   @param confFile is called "config.xml" by default. We avoid the convention
   of dot in the beginning as it might have problem in Windows.

*/
int Configuration::readConfiguration(std::string confFile)
{
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
//        parser.set_validate();
        parser.set_substitute_entities(); //We just want the text to be resolved/unescaped automatically.
        parser.parse_file(confFile);
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
        return 0;        
    }
#endif // HAS_LIBXML
    properties[AUTOSCHEDULE] = PropertyReader::getProperty(confFile, AUTOSCHEDULE);
    properties[CREATESOLVER] = PropertyReader::getProperty(confFile, CREATESOLVER);;
	return 1;
}


Configuration::Configuration(std::string confFile)
{
    properties[AUTOSCHEDULE] = "true";
    properties[CREATESOLVER] = "true";
    
    readConfiguration(confFile);
}
extern int testTrim();

// int main(void)
// {
//     testTrim();
    
//     Configuration("config.xml");
//     return 0;    
// }
#endif //_CONFIGURATION_CPP
