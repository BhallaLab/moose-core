/***
 *       Filename:  test_LSODE.cpp
 *
 *    Description:  Test script for LSODE.
 *
 *        Version:  0.0.1
 *        Created:  2018-08-09

 *       Revision:  none
 *
 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *
 *        License:  GNU GPL2
 */

#include "LSODA.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include "../utility/testing_macros.hpp"

using namespace std;

// Describe the system.
static void fex(double t, double *y, double *ydot, void *data)
{
    ydot[0] = 1.0E4 * y[1] * y[2] - .04E0 * y[0];
    // Don't swap ydot[1] and ydot[2]. The order will change and test will fail.
    ydot[2] = 3.0E7 * y[1] * y[1];
    ydot[1] = -1.0 * (ydot[0] + ydot[2]);
}

static void system_scipy( double t, double* y, double* ydot, void* data)
{
    double mu = 1E4;
    ydot[0] = y[1];
    ydot[1] = mu * (1- y[0]*y[0])*y[1] - y[0];
}

int test_scipy_sys( void )
{
    cout << "Running test scipy sys" << endl;
    double t, tout;
    t = 0e0; tout = 10;

    array<double, 2> y = {10,0};
    int istate = 1;

    LSODA lsoda;

    vector<double> yout;
    lsoda.lsoda_update( system_scipy, 2, &y[0], yout, &t, tout, &istate, nullptr );
    printf(" at t= %12.4e y= %14.6e %14.6e\n", t, yout[1], yout[2]);
    lsoda.n_lsoda_terminate();

    if (istate <= 0)
    {
        cerr << "error istate = " <<  istate << endl;
        exit(0);
    }
    return 0;
}

int test_fex(void)
{
    cout << "Running test fex." << endl;
    int neq = 3;
    double t, tout, y[4];
    t = 0e0;
    tout = 0.4e0;
    y[0] = 1e0;
    y[1] = 0e0;
    y[2] = 0.0;
    int istate = 1;

    LSODA lsoda;
    setprecision( 12 );

    vector<double> res;

    vector<double> yout;
    for (size_t iout = 1; iout <= 12; iout++)
    {
        lsoda.lsoda_update( fex, neq, y, yout, &t, tout, &istate, nullptr );
        // printf(" at t= %12.4e y= %14.6e %14.6e %14.6e\n", t, y[0], y[1], y[2]);
        cerr << " at t " << t << " y= " << yout[1] << ' ' << yout[2] << ' '
            << yout[3] << endl;

        // Update the y for next iteration.
        y[0] = yout[1];
        y[1] = yout[2];
        y[2] = yout[3];

        res.push_back( y[0] );
        res.push_back( y[1] );
        res.push_back( y[2] );

        if (istate <= 0)
        {
            cerr << "error istate = " <<  istate << endl;
            exit(0);
        }
        tout = tout * 10.0E0;
    }

    lsoda.n_lsoda_terminate();

    vector<double> expected = {
        9.851712e-01,  3.386380e-05, 1.479493e-02
        , 9.055333e-01, 2.240655e-05, 9.444430e-02
        , 7.158403e-01, 9.186334e-06, 2.841505e-01
        , 4.505250e-01, 3.222964e-06, 5.494717e-01
        , 1.831976e-01, 8.941773e-07, 8.168015e-01
        , 3.898729e-02, 1.621940e-07, 9.610125e-01
        , 4.936362e-03, 1.984221e-08, 9.950636e-01
        , 5.161833e-04, 2.065787e-09, 9.994838e-01
        , 5.179804e-05, 2.072027e-10, 9.999482e-01
        , 5.283675e-06, 2.113481e-11, 9.999947e-01
        , 4.658667e-07, 1.863468e-12, 9.999995e-01
        , 1.431100e-08, 5.724404e-14, 1.000000e+00
    };

    // Assert here.
    for (size_t i = 0; i < expected.size(); i++)
        ASSERT_DOUBLE_EQ( "LSODE", expected[i], res[i] );

    cout << endl << "The correct answer (up to certain precision): \n"
             " at t=   4.0000e-01 y=   9.851712e-01   3.386380e-05   1.479493e-02 \n"
             " at t=   4.0000e+00 y=   9.055333e-01   2.240655e-05   9.444430e-02 \n"
             " at t=   4.0000e+01 y=   7.158403e-01   9.186334e-06   2.841505e-01 \n"
             " at t=   4.0000e+02 y=   4.505250e-01   3.222964e-06   5.494717e-01 \n"
             " at t=   4.0000e+03 y=   1.831976e-01   8.941773e-07   8.168015e-01 \n"
             " at t=   4.0000e+04 y=   3.898729e-02   1.621940e-07   9.610125e-01 \n"
             " at t=   4.0000e+05 y=   4.936362e-03   1.984221e-08   9.950636e-01 \n"
             " at t=   4.0000e+06 y=   5.161833e-04   2.065787e-09   9.994838e-01 \n"
             " at t=   4.0000e+07 y=   5.179804e-05   2.072027e-10   9.999482e-01 \n"
             " at t=   4.0000e+08 y=   5.283675e-06   2.113481e-11   9.999947e-01 \n"
             " at t=   4.0000e+09 y=   4.658667e-07   1.863468e-12   9.999995e-01 \n"
             " at t=   4.0000e+10 y=   1.431100e-08   5.724404e-14   1.000000e+00 \n"
             << endl;
    return 0;
}

int main(int argc, const char *argv[])
{
    test_scipy_sys();
    test_fex();
    return 0;
}
