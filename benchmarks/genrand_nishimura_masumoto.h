/***
 *       Filename:  genrand_nishimura_masumoto.h
 *
 *    Description:  Mersenne twister by Takuji Nishimura and Makoto Matsumoto.
 *                  This is here for benchmarking purpose.
 *
 *        Version:  0.0.1
 *        Created:  2016-03-31
 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */

#ifndef GENRAND_NISHIMURA_MASUMOTO_H_JFLWCH6H
#define GENRAND_NISHIMURA_MASUMOTO_H_JFLWCH6H

namespace tnmm
{

/* Period parameters */
#define _N 624
#define _M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[_N]; /* the array for the state vector  */
static int mti=_N+1; /* mti==N+1 means mt[N] is not initialized */
unsigned long genrand_int32(void);
void init_by_array(unsigned long init_key[], int key_length);

}

#endif /* end of include guard: GENRAND_NISHIMURA_MASUMOTO_H_JFLWCH6H */

