/*******************************************************************
 * File:            ArgParser.cpp
 * Description:     This is the implementation of ArgParser class.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-01-03 15:56:32
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

#ifndef _ARGPARSER_CPP
#define _ARGPARSER_CPP

#include <cstdlib> // Required for g++ 4.3.2
#include "ArgParser.h"

map <char, string > ArgParser::option_;
map <string, char> ArgParser::longShortMap_;
vector <string> ArgParser::scriptArgs_;

    
/**
   Parse command line arguments
   @return 0 if success,
   1 if error
   print help message and exit if -h or --help is passed.
*/
int ArgParser::parseArguments(int argc, char **argv)    
{
    static bool inited = false;
    if (!inited)
    {
        
        /**
           these are the defined command line parameters, the actual
           command line will have these characters preceded by '-',
           or in the long format that starts with '--'.
        */
        
        option_['c'] = "moose.prop"; // configuration file
        longShortMap_["config-file"] = 'c';
        option_['d'] = ""; // DOCPATH 
        longShortMap_["docpath"] = 'd';
        option_['h'] =
            "This is MOOSE : the Messaging Object Oriented Simulation Environment,\n"
            + string("You can run it as \n\t")
            + string(argv[0])
            + string(" [OPTION1 OPTION2 ... ] [SCRIPT_FILE]\n where OPTION_S can be any of the following -\n")
            + string("\t-c, --config-file <config_file>  Read configuration from config_file\n")
            + string("\t-d, --docpath <doc_dir>          Search doc_dir for documentation\n")
            + string("\t-h, --help                       Show this help message and exit\n")
            + string("\t-p, --simpath <sim_path>         Search sim_path for include files.\n")
            + string("See MOOSE Documentation for further information (http://moose.sourceforge.net or http://moose.ncbs.res.in)");
        longShortMap_["help"] = 'h';
        
        option_['p'] = ""; // SIMPATH
        longShortMap_["simpath"] = 'p';
    }
    if (( argc == 0 ) || ( argv == NULL ) )
    {
        return 0;
    }
    
    for ( int i = 1; i < argc; ++ i )
    {
        char opt = 0;
            
        string argument(argv[i]);
            
        if (argument[0] == '-')
        {
            if ( argument.length() > 2 )
            {
                if ( argument[1] == '-' ) // long format
                {
                    map <string, char>::iterator it = longShortMap_.find(argument.substr(2));
                    if ( it == longShortMap_.end())
                    {
                        cerr << "ERROR: invalid argument " << argv[i] << endl
                             << "To get a list of allowed arguments, run MOOSE as: " << argv[0] << " -h" << endl;
                        return 1;
                    }
                    opt = it->second;
                }
                else
                {
                    cerr << "ERROR: invalid argument " << argv[i] << endl
                         << "To get a list of allowed arguments, run MOOSE as: " << argv[0] << " -h" << endl;
                    return 1;
                }                        
            }
            else if ( argument.length() == 2 ) // expecting short format
            {
                opt = argument[1];
            }
            else
            {
                cerr << "ERROR: invalid argument " << argv[i] << endl
                     << "To get a list of allowed arguments, run MOOSE as: " << argv[0] << " -h" << endl;
                return 1;
            }
            map <char, string>::iterator it = option_.find(opt);
            if ( it == option_.end())
            {
                cerr << "ERROR: invalid argument " << argv[i] << endl
                     << "To get a list of allowed arguments, run MOOSE as: " << argv[0] << " -h" << endl;
                return 1;
            }
            if (opt == 'h')
            {
                cout << option_['h'] << endl;
                exit( 0 );
            }
            
            ++i;
            if ( i >= argc ) {
                cerr << "ERROR: No value for parameter " << argv[i-1] << endl
                     << "To get a list of allowed arguments, run MOOSE as: " << argv[0] << " -h" << endl;
                return 1;                
            }
            string value(argv[i]);
            option_[opt] = value;                
        } // end if (argument[0] == '-')
        else{
            // first unpaired option starting with anything but '-'
            // is considered the script to be run and all the
            // arguments following it arguments to the MOOSE script
            while ( ++i < argc )
            {
                scriptArgs_.push_back(argument);     
                argument = string(argv[i]);
            }
            scriptArgs_.push_back(argument);
        }            
    } // end for
        
    return 0;
} // end parseArguments

string ArgParser::getConfigFile()
{
    map<char, string> :: const_iterator i = option_.find('c');
    return ( i != option_.end() )? i->second : "";    
}
string ArgParser::getSimPath()
{
    map<char, string> :: const_iterator i = option_.find('p');
    return ( i != option_.end() )? i->second : "";    
}
string ArgParser::getDocPath()
{    
    map<char, string> :: const_iterator i = option_.find('d');
    return ( i != option_.end() )? i->second : "";    
}
/**
   Get the list of script and the arguments to it
*/
const vector<string> ArgParser::getScriptArgs()
{
    return scriptArgs_;
}

#ifdef TEST_ARGPARSER // test main

int main(int argc, char **argv)
{
    ArgParser::parseArguments(argc, argv);
    cout << "SIMPATH = " << ArgParser::getSimPath() << endl
         << "DOCPATH = " << ArgParser::getDocPath() << endl
         << "CONFIG = " << ArgParser::getConfigFile() << endl
         << "SCRIPT ARGUMENTS = " << endl;
    vector <string> scriptArgs = ArgParser::getScriptArgs();
    
    for ( unsigned int i = 0; i < scriptArgs.size(); ++i )
    {
        cout << scriptArgs[i] << endl;
    }
    return 0;    
}

#endif    // TEST_ARGPARSER


#endif
