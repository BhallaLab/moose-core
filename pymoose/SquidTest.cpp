/*******************************************************************
 * File:            SquidTest.cpp
 * Description:     This is a C++ version of squid demo - for
 *                      debugging pymoose ( it will be horrible to
 *                      try debugging the python interpreter running
 *                      squid.py
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-07-04 17:08:44
 ********************************************************************/
#include "pymoose.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

double VMIN = -0.1;
double VMAX = 0.05;

int NDIVS = 150;

double v = VMIN;

double dv = ( VMAX - VMIN ) / NDIVS;

double SIMDT = 1e-5;

double PLOTDT = 1e-4;

double RUNTIME = 0.5;

double EREST = -0.07;

double VLEAK = EREST + 0.010613;

double VK = EREST -0.012;
double VNa = EREST + 0.115;
double RM = 424.4e3;
double RA = 7639.44e3;
double GLEAK = 0.3e-3;
// double GK = 36e-3;
// double GNa = 120e-3;
double CM = 0.007854e-6;
double INJECT = 0.1e-6;

double GK = 0.282743e-3;
double GNa = 0.94248e-3;

double calc_Na_m_A(double v)
{
    if (fabs(EREST+0.025-v) < 1e-6)
    {
        v = v + 1e-6;
    }
    return 0.1e6 * (EREST + 0.025 -v)/(exp((EREST + 0.025 - v)/0.01) - 1.0);            
}


double calc_Na_m_B(double v){

    return 4.0e3 * exp((EREST - v)/0.018);
}




double calc_Na_h_A( double v )
{
    return  70.0 * exp(( EREST - v )/0.020 );
}




double calc_Na_h_B( double v )
{
    
    return ( 1.0e3 / (exp ( ( 0.030 + EREST - v )/ 0.01 ) + 1.0 ));
    
}


    
double calc_K_n_A( double v )
{
    
    if (fabs( 0.01 + EREST - v )  < 1.0e-6 )
    {
        v = v + 1.0e-6;
    }
    return ( 1.0e4 * ( 0.01 + EREST - v ) )/(exp(( 0.01 + EREST - v )/0.01) - 1.0 );
}




double calc_K_n_B( double v )
{
    
    return 0.125e3 * exp((EREST - v ) / 0.08 );
}




int main(int argc, char **argv)
{
    Compartment squid("squid");
    squid.__set_Rm(RM);
    squid.__set_Ra(RA);
    squid.__set_Cm(CM);
    squid.__set_Em(VLEAK);

    HHChannel Na("Na", &squid);
    Na.__set_Ek(VNa);
    Na.__set_Gbar(GNa);
    Na.__set_Xpower(3);
    Na.__set_Ypower(1);

    HHChannel K("K",&squid);
    K.__set_Ek(VK);
    K.__set_Gbar(GK);
    K.__set_Xpower(4);

    squid.connect("channel", &Na, "channel");
    squid.connect("channel", &K, "channel");
    Table Vm("Vm");
    Vm.__set_step_mode(3);
    Vm.connect("inputRequest",&squid,"Vm");
    HHGate Na_xGate(Id(Na.path()+"/xGate"));
    Interpol Na_xGate_A(PyMooseBase::pathToId(Na_xGate.path()+"/A"));
    Interpol Na_xGate_B(PyMooseBase::pathToId(Na_xGate.path()+"/B"));
    HHGate Na_yGate(PyMooseBase::pathToId(Na.path()+"/yGate"));
    Interpol Na_yGate_A (PyMooseBase::pathToId(Na_yGate.path()+"/A"));
    Interpol Na_yGate_B (PyMooseBase::pathToId(Na_yGate.path()+"/B"));
    HHGate K_xGate(Id(K.path()+"/xGate"));
    Interpol K_xGate_A(PyMooseBase::pathToId(K_xGate.path()+"/A"));
    Interpol K_xGate_B(PyMooseBase::pathToId(K_xGate.path()+"/B"));

    Na_xGate_A.__set_xmin(VMIN);
    Na_xGate_B.__set_xmin(VMIN);
    Na_yGate_A.__set_xmin(VMIN);
    Na_yGate_B.__set_xmin(VMIN);
    K_xGate_A.__set_xmin(VMIN);
    K_xGate_B.__set_xmin(VMIN);
    Na_xGate_A.__set_xmax(VMAX);
    Na_xGate_B.__set_xmax(VMAX);
    Na_yGate_A.__set_xmax(VMAX);
    Na_yGate_B.__set_xmax(VMAX);
    K_xGate_A.__set_xmax(VMAX);
    K_xGate_B.__set_xmax(VMAX);
    Na_xGate_A.__set_xdivs( NDIVS);    
    Na_xGate_B.__set_xdivs(NDIVS);
    Na_yGate_A.__set_xdivs(NDIVS);    
    Na_yGate_B.__set_xdivs(NDIVS);    
    K_xGate_A.__set_xdivs(NDIVS);    
    K_xGate_B.__set_xdivs(NDIVS);
    double v = VMIN;
    for( int i = 0; i < Na_xGate_A.__len__(); ++i)
    {
        Na_xGate_A.__setitem__(i, calc_Na_m_A ( v ));
        Na_xGate_B.__setitem__(i, calc_Na_m_A (v)   +  calc_Na_m_B ( v   ));
        
	Na_yGate_A.__setitem__(i, calc_Na_h_A  (v ));
        Na_yGate_B.__setitem__(i,  calc_Na_h_A  (v)   +   calc_Na_h_B  (v   ));
        
	K_xGate_A.__setitem__(i, calc_K_n_A  (v ));
        K_xGate_B.__setitem__(i, calc_K_n_A ( v)   +  calc_K_n_B ( v ));

        v = v + dv;
    }
    
    PyMooseContext context = *(PyMooseBase::getContext());    
    context.setClock(0, SIMDT, 0);    
    context.setClock(1, PLOTDT, 0);    
    context.useClock(PyMooseBase::pathToId("/sched/cj/t0"), "/Vm,/squid,/squid/#");
    
    squid.__set_initVm( EREST );    
    context.reset();    
    squid.__set_inject(0);
    
    context.step(0.005);
    
    squid.__set_inject(INJECT);
    context.step(0.040);
    squid.__set_inject(0);
    context.step(0.005);    
    Vm.dumpFile("squid.plot");
    return 0;
 
}

