/*******************************************************************
 * File:            Configuration.h
 * Description:     This class stores the current configuration
 *                  information. 
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-09-25 08:19:44
 ********************************************************************/
#ifndef _COFIGURATION_H
#define _COFIGURATION_H
#include <map>
#include <string>
using namespace std;

/**
   This class is the configuration manager. Right now it has just a
   map to store the properties. It will be enhanced as required. It
   is very much after the .prop files frequently used in java based
   applications.
   The things to implement:
    Move autoscheduling to configuration file. It will be on by default.
    Path handling.
 */
class Configuration
{
  public:
    /**
       These are the static list of allowed configuration properties.
       These are the keys in the map.
    */
    static const std::string AUTOSCHEDULE;
    static const std::string CREATESOLVER;
    
    Configuration(const std::string configFile="config.xml");
    map <std::string,std::string> properties;
    int readConfiguration(std::string fileName);
};
#endif
