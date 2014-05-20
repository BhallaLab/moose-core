/*
 * ==============================================================================
 *
 *       Filename:  testing_macros.hpp
 *
 *    Description:  This file contains some macros useful in testing. 
 *
 *        Version:  1.0
 *        Created:  Monday 19 May 2014 05:04:41  IST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawar@ee.iitb.ac.in
 *   Organization:  
 *
 * ==============================================================================
 */

#ifndef  TESTING_MACROS_INC
#define  TESTING_MACROS_INC


#include <sstream>
#include <iostream>
#include <exception>
#include "current_function.hpp"
#include "print_function.h"

using namespace std;

class FatalTestFailure 
{
    public:
        FatalTestFailure()
        {
            msg = string("");
        }

        FatalTestFailure(string msg)
        {
            msg = msg;
        }

    public:
        string msg;
};

#define LOCATION(ss) \
    ss << "In function: " << MOOSE_CURRENT_FUNCTION; \
    ss << " file: " << __FILENAME__ << ":" << __LINE__ << endl;  

#define EXPECT_TRUE( condition, msg) \
    if( !(condition) ) {\
        ostringstream ss; \
        ss << "[FAILED] " << msg << endl; \
        cout << ss.str(); \
    }

#define EXPECT_FALSE( condition, msg) \
    if( (condition) ) {\
        ostringstream ss; \
        ss << "[FAILED] " << msg << endl; \
        cout << ss.str(); \
    }

#define EXPECT_EQ(a, b, token)  \
    if( (a) != (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Expected " << a << ", received " << b  << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define EXPECT_NEQ(a, b, token)  \
    if( (a) == (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Not expected " << a << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define EXPECT_GT(a, b, token)  \
    if( (a) <= (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Expected greater than " << b << ", received " << a << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define EXPECT_GTE(a, b, token)  \
    if( (a) < (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Expected greater than or equal to " << b << ", received " << a << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define EXPECT_LT(a, b, token)  \
    if( (a) >= (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Expected less than " << (b) << ", received " << (a) << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define EXPECT_LTE(a, b, token)  \
    if( (a) > (b)) { \
        ostringstream ss; \
        LOCATION(ss); \
        ss << "Expected less than or equal to " << b << ", received " << a << endl; \
        ss << token << endl; \
        dump(ss.str(), "EXPECT_FAILURE"); \
    }

#define ASSERT_TRUE( condition, msg) \
    if( !(condition) ) {\
        ostringstream ss; \
        ss << "[FAILED] " << msg << endl; \
        cout << ss.str(); \
        throw FatalTestFailure(ss.str());  \
    }

#define ASSERT_FALSE( condition, msg) \
    if( (condition) ) {\
        ostringstream ss; \
        ss << "[FAILED] " << msg << endl; \
        cout << ss.str(); \
        throw FatalTestFailure(ss.str()); \
    }


#endif   /* ----- #ifndef TESTING_MACROS_INC  ----- */
