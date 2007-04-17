/*******************************************************************
 * File:            Compartment.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-03-24 06:50:11
 ********************************************************************/

#ifndef _PYMOOSE_COMPARTMENT_H
#define _PYMOOSE_COMPARTMENT_H

class Compartment: public PyMooseBase
{
  public:
    static const std::string className;
    Compartment(Id id);
    Compartment(std::string path);
    
    Compartment(std::string name, unsigned int parentId);
    Compartment(std::string name, PyMooseBase *parent);
    ~Compartment();
    
    const std::string& getType();
    
    void __set_Vm(double vm);
    void __set_Ra(double ra);
    void __set_Rm(double rm);
    double __get_Vm() const; 
    double __get_Ra() const;
    double __get_Rm() const;
    void __set_Em( double Em );
    double __get_Em() const;
    void __set_Cm( double Cm );
    double __get_Cm() const;
    void __set_Im( double Im );
    double __get_Im() const;
    void __set_inject( double inject );
    double __get_inject() const;
    void __set_initVm( double initVm );
    double __get_initVm() const;
    void __set_diameter( double diameter );
    double __get_diameter() const;
    void __set_length( double length );
    double __get_length() const;
#ifdef DO_UNIT_TESTS
    static bool testCompartment(int count, bool doPrint);    
#endif // DO_UNIT_TESTS
    
}; 
#endif // _PYMOOSE_COMPARTMENT_H
