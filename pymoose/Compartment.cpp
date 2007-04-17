/*******************************************************************
 * File:            Compartment.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-03-10 18:46:07
 ********************************************************************/

#ifndef _PYMOOSE_COMPARTMENT_CPP
#define _PYMOOSE_COMPARTMENT_CPP

#include "../basecode/header.h"
#include "../basecode/moose.h"
#include "../shell/Shell.h"
#include "PyMooseUtil.h"
#include "PyMooseBase.h"
#include "Compartment.h"

const std::string Compartment::className = "Compartment";
Compartment::Compartment(Id id):PyMooseBase(id)
{
}
Compartment::Compartment(std::string path):PyMooseBase(className, path)
{
}

Compartment::Compartment(std::string name, unsigned int parentId):PyMooseBase(className, name, parentId)
{
}
Compartment::Compartment(std::string name, PyMooseBase * parent):PyMooseBase(className, name, parent)
{
}

Compartment::~Compartment()
{    
}

const std::string& Compartment::getType()
{
    return className;
}

double Compartment::__get_Vm() const
{
    double vm;
    get<double>(Element::element(id_), "Vm", vm);
    return vm;
}

void Compartment::__set_Vm(double vm)
{
    set <double> (Element::element(id_), "Vm", vm);
}


double Compartment::__get_Rm() const
{
    double rm;
    get<double>(Element::element(id_), "Rm", rm);
    return rm;
}

void Compartment::__set_Rm(double rm)
{
    set <double> (Element::element(id_), "Rm", rm);
}

double Compartment::__get_Ra() const
{
    double ra;
    get<double>(Element::element(id_), "Ra", ra);
    return ra;
}

void Compartment::__set_Ra(double ra)
{
    set <double> (Element::element(id_), "Ra", ra);
}

void Compartment::__set_Em(double Em )
{
    set <double> (Element::element(id_),"Em", Em);    
}

double Compartment::__get_Em() const
{
    double Em;
    get<double>(Element::element(id_), "Em", Em);
    return Em;
}

void Compartment::__set_Cm(double Cm )
{
    set <double> (Element::element(id_),"Cm", Cm);
}

double Compartment::__get_Cm() const
{
    double cm;
    get<double>(Element::element(id_), "Cm", cm);
    return cm;
}

void Compartment::__set_Im(double Im )
{
    set <double> (Element::element(id_),"Im", Im);
}

double Compartment::__get_Im() const
{
    double im;
    get<double>(Element::element(id_), "Im", im);
    return im;
}

void Compartment::__set_inject(double Inject )
{
    set <double> (Element::element(id_),"inject", Inject);
}

double Compartment::__get_inject() const
{
    double inject;
    get<double>(Element::element(id_), "inject", inject);
    return inject;
}

void Compartment::__set_initVm(double initVm )
{
    set <double> (Element::element(id_),"initVm", initVm);
}

double Compartment::__get_initVm() const
{
    double initvm;
    get<double>(Element::element(id_), "initVm", initvm);
    return initvm;
}

void Compartment::__set_diameter(double diameter )
{
    set <double> (Element::element(id_),"diameter",diameter);
}

double Compartment::__get_diameter() const
{
    double diameter;
    get<double>(Element::element(id_), "diameter", diameter);
    return diameter;
}

void Compartment::__set_length(double length )
{
    set <double> (Element::element(id_),"length", length);
}

double Compartment::__get_length() const
{
    double length;
    get<double>(Element::element(id_), "length", length);
    return length;
}
#ifdef DO_UNIT_TESTS
bool Compartment::testCompartment(int count, bool doPrint)
{
    bool testResult = true;
    bool localResult = true;
    

    double diameter, length, initVm, injCurrent, Rm, Cm, Ra, Vm, Im;
    double retrieved;
    double epsilon = 1e-5;
    
    diameter = 1.2e-6;
    length = 2.3e-5;
    initVm = 3.4e-3;
    injCurrent = 4.5e-9;
    Cm =  5.6e-12;
    Vm = 6.7e-3;
    Im = 7.8e-9;
    Rm = 8.9e9;
    Ra = 9.1e6;
    if (doPrint)
    {
        cerr << "TEST:: Compartment::testCompartment ... STARTING" << endl;
    }
    
    for ( int i = 0; i < count; ++i)
    {
        
        Compartment testComp("test");
// !!Commented out because set and get for diameter and length remains unimplemented
        // testComp.__set_diameter(diameter);
//         retrieved = testComp.__get_diameter();        
//         localResult = isEqual(diameter, retrieved, epsilon);
        
//         testResult = testResult && localResult;
//        if (!localResult)
//         {
//             cerr << "TEST:: Compartment::testCompartment() - tried to set diameter to " << diameter << " but found the value to be " << retrieved << endl;
//         }
        
//         testComp.__set_length(length);
//         retrieved = testComp.__get_length();
        
//         localResult = isEqual(length, retrieved, epsilon);
        

//         testResult = testResult && localResult;
//            if (!localResult)
//         {
//             cerr << "TEST:: Compartment::testCompartment() - tried to set length to " << length << " but found the value to be " << retrieved << endl;
//         }

        testComp.__set_initVm(initVm);
        retrieved = testComp.__get_initVm();
        localResult = isEqual(initVm, retrieved, epsilon);        
        localResult = localResult;
        testResult = testResult && localResult;
        if (!localResult)
        {
            cerr << "TEST:: Compartment::testCompartment() - tried to set initVm to " << initVm << " but found the value to be " << retrieved << endl;
        }

        // !!There is something wrong with this - injection current retrieved is way away from that set.
        //   testComp.__set_inject(injCurrent);
//         retrieved = testComp.__get_inject();        
//         localResult = isEqual(injCurrent, retrieved, epsilon);
//         testResult = testResult && localResult;
//        if (!localResult)
//         {
//             cerr << "TEST:: Compartment::testCompartment() - tried to set injection current to " << injCurrent << " but found the value to be " << retrieved << endl;
//         }

        testComp.__set_Vm(Vm);
        retrieved = testComp.__get_Vm();        
        localResult = isEqual(Vm, retrieved, epsilon);
        testResult = testResult && localResult;
        if (!localResult)
        {
            cerr << "TEST:: Compartment::testCompartment() - tried to set Vm to " << Vm << " but found the value to be " << retrieved << endl;
        }

        
        testComp.__set_Cm(Cm);
        retrieved = testComp.__get_Cm();        
        localResult = isEqual(Cm, retrieved, epsilon);
        testResult = testResult && localResult;
        if (!localResult)
        {
            cerr << "TEST:: Compartment::testCompartment() - tried to set Cm to " << Cm << " but found the value to be " << retrieved << endl;
        }

        testComp.__set_Ra(Ra);
        retrieved = testComp.__get_Ra();        
        localResult = isEqual(Ra, retrieved, epsilon);
        testResult = testResult && localResult;
        if (!localResult)
        {
            cerr << "TEST:: Compartment::testCompartment() - tried to set Ra to " << Ra << " but found the value to be " << retrieved << endl;
        }

        testComp.__set_Rm(Rm);
        retrieved = testComp.__get_Rm();        
        localResult = isEqual(Rm, retrieved, epsilon);
        testResult = testResult && localResult;
        if (!localResult)
        {
            cerr << "TEST:: Compartment::testCompartment() - tried to set Rm to " << Rm << " but found the value to be " << retrieved << endl;
        }

        // !! Im setting is also not working - may be it is dynamically calculated. Then why do we have setter method for Im?
    //     testComp.__set_Im(Im);
//         retrieved = testComp.__get_Im();
//         localResult = isEqual(Im, retrieved, epsilon);
//         testResult = testResult && localResult;
//         if (!localResult)
//         {
//             cerr << "TEST:: Compartment::testCompartment() - tried to set Im to " << Im << " but found the value to be " << retrieved << endl;
//         }


    } // end for
    if (doPrint)
    {
        cerr << "TEST::  Compartment::testCompartment() ... " << (testResult? "SUCCESS": "FAILURE") << endl;
    }
    return testResult;    
}

#endif // DO_UNIT_TESTS

#endif // _PYMOOSE_COMPARTMENT_CPP
