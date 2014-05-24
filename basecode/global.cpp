/*
 * ==============================================================================
 *
 *       Filename:  global.cpp
 *
 *    Description:  It contains global variables to track no of test run and
 *    running performance of moose basecode.
 *
 *        Version:  1.0
 *        Created:  Tuesday 29 April 2014 10:18:35  IST
 *       Revision:  0.1
 *       Compiler:  gcc/g++
 *
 *         Author:  Dilawar Singh 
 *   Organization:  Bhalla's lab, NCBS Bangalore
 *
 * ==============================================================================
 */

#include "global.h"
#include "../external/debug/simple_logger.hpp"
#include <numeric>

/*-----------------------------------------------------------------------------
 *  This variable keep track of how many tests have been performed.
 *
 *  Check header.h for macros tbegin and tend which uses it.
 *-----------------------------------------------------------------------------*/
unsigned int totalTests = 0;

stringstream errorSS;

clock_t simClock = clock();

extern int checkPath( const string& path);
extern string joinPath( string pathA, string pathB);
extern string fixPath( string path);
extern string dumpStats( int  );

/* Logger */
SimpleLogger logger;

namespace moose {
    /* Check if path is OK */
    int checkPath( const string& path  )
    {
        if( path.size() < 1)
            return EMPTY_PATH;

        if( path.find_first_of( " \\!") != std::string::npos )
            return BAD_CHARACTER_IN_PATH;

        if ( path[path.size() - 1 ] != ']')
        {
            return MISSING_BRACKET_AT_END;
        }
        return 0;
    }

    /* Join paths */
    string joinPath( string pathA, string pathB )
    {
        errorSS.str("");
        errorSS << "Calling a hacky function to fix paths. Ticket #134"
            << endl;
        dump(errorSS.str(), "BUG");
        pathA = moose::fixPath( pathA );
        string newPath = pathA + "/" + pathB;
        return moose::fixPath( newPath );
    }

    /* Fix given path */
    string fixPath(string path)
    {
        int pathOk = moose::checkPath( path );
        if( pathOk == 0)
            return path;
        else if( pathOk == MISSING_BRACKET_AT_END)
            return path + "[0]";
        dump("I don't know how to fix this path: " + path, "FIXME");
        return path;
    }

}
