// Variable.h ---
// Filename: Variable.h
// Description:
// Author: Subhasis Ray
// Maintainer: Dilawar Singh
// Created: Fri May 30 19:37:24 2014 (+0530)

#ifndef _VARIABLE_H
#define _VARIABLE_H

class ObjId;
class Eref;
class Cinfo;

using namespace std;

/** This class is used as FieldElement in Function. It is used as named
   variable of type double.
 */
class Variable
{

public:

    Variable(): name_(""), value_(0.0)
    {
    };

    Variable(const string& name): name_(name), value_(0.0)
    {
    };

    Variable(const Variable& rhs): name_(rhs.name_), value_(rhs.value_)
    {
        ;
    }

    virtual ~Variable() {};

    void setValue(double v)
    {
        value_ = v;
    }

    virtual void epSetValue(const Eref & e, double v)
    {
        value_ = v;
    }

    double getValue() const
    {
        return value_;
    }

    double* ref() 
    {
        return &value_;
    }

    std::string getName() const
    {
        return name_;
    }

    void addMsgCallback(const Eref& e, const string& finfoName, ObjId msg, unsigned int msgLookup);

    static const Cinfo * initCinfo();

private:
    std::string name_;
    double value_;
};

#endif



//
// Variable.h ends here
