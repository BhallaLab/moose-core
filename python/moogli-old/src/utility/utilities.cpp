#include "utility/utilities.hpp"

/** Copied as it is from gsl source file sys/fcmp.c version 1.16 **/
int
compare_double( const double x1
              , const double x2
              , const double epsilon
              )
{
    int exponent;
    double delta, difference;
    /* Find exponent of largest absolute value */
    {
        double max = (fabs(x1) > fabs(x2)) ? x1 : x2;
        frexp (max, &exponent);
    }

    /* Form a neighborhood of size  2 * delta */
    delta = ldexp (epsilon, exponent);
    difference = x1 - x2;
    if(difference > delta)          /* x1 > x2 */
    {
        return 1;
    }
    else if (difference < -delta)   /* x1 < x2 */
    {
        return -1;
    }
    else                            /* -delta <= difference <= delta */
    {
        return 0;                   /* x1 ~=~ x2 */
    }
}
