/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef CUDAODESYSTEM_H

#define CUDAODESYSTEM_H

#include <string>
#include <iostream>

using namespace std;

class CudaOdeSystem
{

public:
    CudaOdeSystem() : method( "rk5" ), initStepSize( 0.001 )
        , epsAbs( 1e-6 ), epsRel( 1e-6 )
    {
    }

    void print( )
    {
        cout << "+ Ode system " << endl;
        cout << " Method : " << method << endl;
        cout << " Dimentions: " << dimension << endl;
        cout << " State : ";
        for( size_t i = 0; i < dimension; i++ )
            cout << f[i] << ' ';
        cout << endl;
        cout << "------------------------------" << endl;
    }

    std::string method;

    double initStepSize;
    double epsAbs; // Absolute error
    double epsRel; // Relative error

    size_t dimension;

    /**
     * @brief f = y. double* f is similar to the following in gsl-ode example.
     * https://www.gnu.org/software/gsl/manual/html_node/ODE-Example-programs.html
     */
    double* f;
    double* y;

};

#endif /* end of include guard: CUDAODESYSTEM_H */
