#ifndef _PYMOOSE_NEUTRAL_H
#define _PYMOOSE_NEUTRAL_H

#include "../basecode/header.h"
#include "PyMooseBase.h"

class Neutral: public PyMooseBase
{
   public:
    static const std::string className;
    Neutral(Id id);
    Neutral(std::string path); 
    Neutral(std::string name, unsigned int parentId);
    Neutral(std::string name, PyMooseBase* parent);   
    ~Neutral();
    const std::string& getType();   
};
#endif // _PYMOOSE_NEUTRAL_H
   
