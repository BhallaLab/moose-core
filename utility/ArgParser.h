/*******************************************************************
 * File:            ArgParser.h
 * Description:     Contains function to parse the command line
 *                  parameters passed to moose.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-02 20:34:14
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

#ifndef _ARGPARSER_H
#define _ARGPARSER_H

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;
class ArgParser
{
  public:    
    static int parseArguments(int argc, char **argv);    

    static string getConfigFile();    

    static string getSimPath();
    
    static string getDocPath();
    
    static const vector <string> getScriptArgs();
    
  private:
    ArgParser();
    // option, longShortMap and scriptArgs have been converted from
    // static variables to static
    // methods returning reference to the contents of static pointers
    // in order to avoid the static initialization fiasco.
    /** Returns the command line parameter flag - must start with a '-' .. following glib way */
    static map <char, string>& option();
    
    /**
       Returns the map for long options starting with "--" to short options starting with "-".
       Short options are single character.
    */
    static map <string , char>& longShortMap();

    /**
       Returns the script file, which is the last argument passed in
       command line along with the arguments to the script.
    */
    static vector <string>& scriptArgs();

    
};

#endif
