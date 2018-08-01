/*
 * =====================================================================================
 *
 *       Filename:  RNG.h
 *
 *    Description:  Random Number Generator class
 *
 *        Created:  05/09/2016 12:00:05 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */


#ifndef  __RNG_INC
#define  __RNG_INC

#ifdef  USE_BOOST_RNG

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>

#if  defined(BOOST_RANDOM_DEVICE_EXISTS)
#include <boost/random/random_device.hpp>
#endif  // BOOST_RANDOM_DEVICE_EXISTS

#endif     /* -----  not USE_BOOST_RNG  ----- */

#include <limits>
#include <iostream>
#include "randnum.h"
#include <random>

using namespace std;

namespace moose {

/*
 * =====================================================================================
 *        Class:  RNG
 *  Description:  Random number generator class.
 * =====================================================================================
 */

template < typename T >
class RNG
{
    public:
        RNG ()                                  /* constructor      */
        {
            // Setup a random seed if possible.
            setRandomSeed( );
        }

        ~RNG ()                                 /* destructor       */
        { ; }

        void setRandomSeed( )
        {

#ifdef USE_BOOST_RNG
#ifdef BOOST_RANDOM_DEVICE_EXISTS
            boost::random::random_device rd;
#else
            std::random_device rd;
#endif                                // BOOST_RANDOM_DEVICE_EXISTS
#else                                 // USE C++11
            std::random_device rd;
#endif
            setSeed( rd() );
        }

        /* ====================  ACCESSORS     ======================================= */
        T getSeed( void )
        {
            return seed_;
        }

        /* ====================  MUTATORS      ======================================= */
        /**
         * @brief If seed if 0 then set seed to a random number else set seed to
         * the given number.
         *
         * @param seed
         */
        void setSeed( const unsigned long int seed )
        {
            seed_ = seed;
            if( seed == 0 )
            {
                setRandomSeed( );
                return;
            }

            rng_.seed( seed_ );
        }

        /**
         * @brief Generate a uniformly distributed random number between a and b.
         *
         * @param a Lower limit (inclusive)
         * @param b Upper limit (inclusive).
         */
        T uniform( const T a, const T b)
        {
            return ( b - a ) * dist_( rng_ ) + a;
        }

        /**
         * @brief Return a uniformly distributed random number between 0 and 1
         * (exclusive).
         *
         * @return randum number.
         */
        T uniform( void )
        {
            return dist_( rng_ );
        }


    private:
        /* ====================  DATA MEMBERS  ======================================= */
        T res_;
        T seed_;

        // By default use <random>.
#if USE_BOOST_RNG
        boost::random::mt19937 rng_;
        boost::random::uniform_01<T> dist_;
#else
        std::mt19937 rng_;
        std::uniform_real_distribution<> dist_;
#endif

}; /* -----  end of template class RNG  ----- */

}                                               /* namespace moose ends  */

#endif   /* ----- #ifndef __RNG_INC  ----- */
