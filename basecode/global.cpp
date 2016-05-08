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


/*-----------------------------------------------------------------------------
 *  This variable keep track of how many tests have been performed.
 *
 *  Check header.h for macros tbegin and tend which uses it.
 *-----------------------------------------------------------------------------*/
unsigned int totalTests = 0;

stringstream errorSS;
std::random_device rd;


bool isRNGInitialized = false;

clock_t simClock = clock();

extern int checkPath( const string& path);
extern string joinPath( string pathA, string pathB);
extern string fixPath( string path);
extern string dumpStats( int  );



namespace moose {

    int __rng_seed__ = rd();

    rng_type_ rng( __rng_seed__ );
    distribution_type_ dist;

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
        return path;
    }

    /**
     * @brief Set the global seed or all rngs.
     *
     * @param x 
     */
    void mtseed( unsigned int x )
    {
        moose::rng.seed( x );
        moose::__rng_seed__ = x;
        isRNGInitialized = true;
    }

    /*  Generate a random number */
    double mtrand( void )
    {
        return moose::dist( rng );
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
     * @brief Create directories recursively needed to open the given file p. 
     *
     * @param path When successfully created, returns created path, else
     * convert path to a filename by replacing '/' by '_'.
     */
    string createParentDirs( string p )
    {
        if( p.size() == 0 )
            return string("");
#ifdef  USE_BOOST
        try 
        {
            boost::filesystem::path pdirs( p );
            pdirs.remove_filename();
            if( pdirs.string().size() == 0 )
                return p;
            boost::filesystem::create_directories( pdirs );
            LOG( moose::info, "Created directory " << p );
        }
        catch(const boost::filesystem::filesystem_error& e)
        {
            LOG( moose::warning, "create_directories(" << p << ") failed with "
                    << e.code().message()
               );
            std::replace(p.begin(), p.end(), '/', '_' );
            std::replace(p.begin(), p.end(), '\\', '_' );
            return p;
        }
#else      /* -----  not USE_BOOST  ----- */
        std::replace(p.begin(), p.end(), '/', '_' );
        std::replace(p.begin(), p.end(), '\\', '_' );
        return p;
#endif     /* -----  not USE_BOOST  ----- */
        return p;
    }


    /*  return extension of a filename */
    string getExtension(const string& path, bool without_dot )
    {
        auto dotPos = path.find_last_of( '.' );
        if( dotPos == std::string::npos )
            return "";

        if( without_dot )
            return path.substr( dotPos + 1 );

        return path.substr( dotPos );
    }

    /*  returns `basename path`  */
    string pathToName( const string& path )
    {
        return path.substr( path.find_last_of( '/' ) );
    }

    /*  /a[0]/b[1]/c[0] -> /a/b/c  */
    string moosePathToUserPath( string path )
    {
        size_t p1 = path.find( '[', 0 );
        while( p1 != std::string::npos )
        {
            size_t p2 = path.find( ']', p1 );
            path.erase( p1, p2-p1+1 );
            p1 = path.find( '[', p2 );
        }
        return path;
    }

    /*  Return formatted string 
     *  Precision is upto 17 decimal points.
     */
    string toString( double x )
    {
        char buffer[50];
        sprintf(buffer, "%.17g", x );
        return string( buffer );
    }
}
