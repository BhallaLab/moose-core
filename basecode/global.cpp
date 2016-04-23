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
#include <numeric>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

#include "../external/debug/simple_logger.hpp"

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

int __rng_seed__ = 0;

/* Logger */
SimpleLogger logger;

/** 
 * @brief Set the global seed for random number generators. 
 *
 * FIXME: When reinit() is * called, each rng should use this value to seed
 * itself, really?
 *
 * @param seed
 */
void mtseed( int seed ) 
{ 
    __rng_seed__ = seed; 
}

/**
 * @brief Global function to generate a random number.
 *
 * @return 
 */
double mtrand( void )
{
    static boost::random::mt19937 rng( __rng_seed__ );
    static boost::random::uniform_01<double> dist;
    return dist( rng );
}

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
