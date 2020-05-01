/***
 *    Description:  interface to random number generator.
 *
 *        Created:  2020-04-03

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  MIT License
 */

#ifndef RANDNUM_H
#define RANDNUM_H

#include <random>
using namespace std;

#include "RNG.h"

namespace moose {

/**
 * @Synopsis  Global RNG.
 */
extern RNG rng;

/**
 * @brief A global seed for all RNGs in moose. When moose.seed( x ) is called,
 * this variable is set. Other's RNGs (except muparser) uses this seed to
 * initialize them. By default it is initialized by random_device (see
 * global.cpp).
 */
extern unsigned long __rng_seed__;

/**
 * @brief Seed seed for RNG.
 *
 * @param seed
 */
void mtseed(unsigned int seed);

/**
 * @brief Generate a random double between 0 and 1
 *
 * @return  A random number between 0 and 1.
 */
double mtrand(void);

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Overloaded function. Random number between a and b
 *
 * @Param a lower limit.
 * @Param b Upper limit.
 *
 * @Returns
 */
/* ----------------------------------------------------------------------------*/
double mtrand(double a, double b);

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Get the global seed set by call of moose.seed( X )
 *
 * @Returns  seed (int).
 */
/* ----------------------------------------------------------------------------*/
int getGlobalSeed();

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Set the seed for all random generator. When seed of a RNG is
 * not set, this seed it used. It is set to -1 by default.
 *
 * @Param seed
 */
/* ----------------------------------------------------------------------------*/
void setGlobalSeed(int seed);
};

#endif /* end of include guard: RANDNUM_H */

