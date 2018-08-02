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

    /* --------------------------------------------------------------------------*/
    /**
     * @Synopsis  Global random number generator engine. Everywhere we use this
     * engine.
     */
    /* ----------------------------------------------------------------------------*/
#ifdef USE_BOOST_RNG
    typedef boost::random::mt19937 MOOSE_RNG_DEFAULT_ENGINE;
#ifdef BOOST_RANDOM_DEVICE_EXISTS
    typedef boost::random::random_device MOOSE_RANDOM_DEVICE;
#else
    typedef std::random_device MOOSE_RANDOM_DEVICE;
#endif
#else
    typedef std::mt19937 MOOSE_RNG_DEFAULT_ENGINE;
    typedef std::random_device MOOSE_RANDOM_DEVICE;
    typedef std::normal_distribution<double> MOOSE_NORMAL_DISTRIBUTION;
#endif

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
            MOOSE_RANDOM_DEVICE rd;
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
        MOOSE_RNG_DEFAULT_ENGINE rng_;

#if USE_BOOST_RNG
        boost::random::uniform_01<T> dist_;
#else
        std::uniform_real_distribution<> dist_;
#endif

}; /* -----  end of template class RNG  ----- */

}                                               /* namespace moose ends  */

#endif   /* ----- #ifndef __RNG_INC  ----- */
