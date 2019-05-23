// Variable.h ---
// Filename: Variable.h
// Description:
// Author: Subhasis Ray
// Maintainer: Dilawar Singh
// Created: Fri May 30 19:37:24 2014 (+0530)

#ifndef _VARIABLE_H
#define _VARIABLE_H

#include "../utility/print_function.hpp"
class ObjId;

/** This class is used as FieldElement in Function. It is used as named
   variable of type double.
 */
class Variable
{

public:

    Variable():value(0.0)
    {
    };

    Variable(const Variable& rhs): value(rhs.value)
    {
        ;
    }

    virtual ~Variable(){};

    void setValue(double v)
    {
        value = v;
    }

    virtual void epSetValue(const Eref & e, double v)
    {
        value = v;
    }

    double getValue() const
    {
        return value;
    }

    void addMsgCallback(const Eref& e, const string& finfoName, ObjId msg, unsigned int msgLookup);

    static const Cinfo * initCinfo();

    double value;
};

#endif



//
// Variable.h ends here
