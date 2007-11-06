/*******************************************************************
 * File:            PropertyReader.h
 * Description:      This class reads a specified property file and
 *                   keeps it as a map.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-08 09:22:06
 ********************************************************************/

#ifndef _PROPERTYREADER_H
#define _PROPERTYREADER_H
#include <string>
#include <vector>
#include <map>
using namespace std;

class PropertyReader
{
  public:
    PropertyReader(std::string fileName);
    ~PropertyReader();
    
    const std::string getProperty(std::string name) const;
    
    const std::string getFileName();    
    vector <std::string> * getPropertyNames();

    static const std::string getProperty(std::string fileName, std::string key);
    static const PropertyReader& getPropertyReader(std::string fileName);
    static void clearAll();    
    static const int MAXCHAR = 1024; /// The maximum no. of characters in a single line of the property file
    
  private:
    map<std::string, std::string> properties_;
    std::string fileName_;
    static map<std::string, PropertyReader*>& getReaders();    
};


#endif
