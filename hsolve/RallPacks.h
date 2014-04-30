/*
 * ==============================================================================
 *
 *       Filename:  RallPacks.h
 *
 *
 *    Description:  RallPacks benchmarks.
 *
 *    This file is part of Moose, the neural simulator. For more details, visit
 *    http://moose.ncbs.res.in .
 *
 *        Version:  1.0
 *        Created:  Wednesday 30 April 2014 01:11:37  IST
 *       Revision:  0.1
 *       Compiler:  g++
 *
 *         Author:  Dilawar Singh, dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * ==============================================================================
 */

#ifndef  RALLPACKS_INC
#define  RALLPACKS_INC

#include "../basecode/header.h"
#include "../biophysics/Compartment.h"

using namespace moose;

/*
 * ==============================================================================
 *        Class:  RallPacks
 *  Description:  
 * ==============================================================================
 */
class RallPacks
{
    
#if  DO_UNIT_TESTS
    friend void runRallpackBenchmarks();
#endif     /* -----  not DO_UNIT_TESTS  ----- */

    public:
        /* ====================  LIFECYCLE     =================================== */
        RallPacks ();                             /* constructor      */
        RallPacks ( const RallPacks &other );   /* copy constructor */
        ~RallPacks ();                            /* destructor       */

        /* ====================  ACCESSORS     =================================== */

        /* ====================  MUTATORS      =================================== */

        /* ====================  OPERATORS     =================================== */

        RallPacks& operator = ( const RallPacks &other ); /* assignment operator */

    protected:
        /* ====================  METHODS       =================================== */

        /* ====================  DATA MEMBERS  =================================== */

    private:
        /* ====================  METHODS       =================================== */

        /* ====================  DATA MEMBERS  =================================== */

}; /* -----  end of class RallPacks  ----- */

#endif   /* ----- #ifndef RALLPACKS_INC  ----- */
