/*
 * ==============================================================================
 *
 *       Filename:  global.cpp
 *
 *    Description:  Some global declarations.
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
#include <random>


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

std::random_device rd;


namespace moose {
    namespace global {

        int __rng_seed__ = rd();

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
            pathA = moose::global::fixPath( pathA );
            string newPath = pathA + "/" + pathB;
            return moose::global::fixPath( newPath );
        }

        /* Fix given path */
        string fixPath(string path)
        {
            int pathOk = moose::global::checkPath( path );
            if( pathOk == 0)
                return path;
            else if( pathOk == MISSING_BRACKET_AT_END)
                return path + "[0]";
            return path;
        }

        /**
         * @brief Set the global seed or all rngs.
         *
         * @param x 
         */
        void mtseed( unsigned int x )
        {
            moose::global::__rng_seed__ = x;
        }

        /*  Generate a random number */
        double mtrand( void )
        {
            static rng_type_ rng( moose::global::__rng_seed__ );
            static distribution_type_ dist;
            return dist( rng );

        }

        // Fix the given path.
        string createPosixPath( string s )
        {
            string undesired = ":?\"<>|[]";
            for (auto it = s.begin() ; it < s.end() ; ++it)
            {
                bool found = undesired.find(*it) != string::npos;
                if(found){
                    *it = '_';
                }
            }
            return s;
        }

        /**
         * @brief Create directories recursively
         *
         * @param path
         */
        void createDirs( boost::filesystem::path p )
        {
            if( p.string().size() == 0 )
                return;
            try 
            {
                boost::filesystem::create_directories( p );
            } 
            catch(const boost::filesystem::filesystem_error& e)
            {
                std::cout << "create_directories(" << p << ") failed with "
                    << e.code().message() << '\n';
            }

        }

    }
}
