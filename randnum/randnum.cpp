/***
 *    Description:  random number generator.
 *
 *        Created:  2020-04-03

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  Same as MOOSE license.
 */

#include "randnum.h"

namespace moose {

unsigned long __rng_seed__ = 0;

RNG rng;

/**
 * @brief Set the global seed or all rngs.
 *
 * @param x
 */
void mtseed(unsigned int x)
{
    static bool isRNGInitialized = false;
    moose::__rng_seed__ = x;
    moose::rng.setSeed(x);
    isRNGInitialized = true;
}

/*  Generate a random number */
double mtrand(void)
{
    return moose::rng.uniform();
}

double mtrand(double a, double b)
{
    return (b - a) * mtrand() + a;
}

int getGlobalSeed()
{
    return __rng_seed__;
}

void setGlobalSeed(int seed)
{
    __rng_seed__ = seed;
}

}  // namespace moose.
