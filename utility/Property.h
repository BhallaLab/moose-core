/*******************************************************************
 * File:            Property.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-04 16:41:14
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

#ifndef _PROPERTY_H
#define _PROPERTY_H
#include <string>
#include <map>
#include <vector>

using namespace std;

#define BUF_SIZE 300

class PathUtility;

class Property
{
  public:
    // property key constants
    static const char* const SIMPATH; // key for path to be searched for script and prototypes
    static const char* const SIMNOTES; // key for notes on MOOSE
    static const char* const DOCPATH; // key for path to be searched for help
    static const char* const AUTOSCHEDULE; // key for autoscheduling on or off
    static const char* const CREATESOLVER; // key for automatic creation of solvers ( on or off )
    static const char* const HOME;
    
    static const int XML_FORMAT;
    static const int PROP_FORMAT;

    static void initialize(string fileName, int format);
    static void initDefaults();
    static string getProperty(string key);
    static void setProperty(string key, string value);
    static int readProperties(string fileName, int format);
    static void readEnvironment();
    static vector <string> & getKeys();
    static void addSimPath(string path);
    static void setSimPath(string paths);
    static const string getSimPath();
    
  private:
    Property();
    static map <string, string>& properties();
    static int readXml(string fileName);
    static int readProp(string fileName);
    static bool initialized_;
    static PathUtility* simpathHandler_;
    
};

    
#endif
