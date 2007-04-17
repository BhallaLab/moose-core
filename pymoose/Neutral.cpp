#ifndef _NEUTRAL_H
#define _NEUTRAL_H

#include "PyMooseBase.h"
#include "Neutral.h"

const std::string Neutral::className = "Neutral";
Neutral::Neutral(Id id):PyMooseBase(id)
{
}

Neutral::Neutral(std::string path):PyMooseBase(className, path)
{
}

Neutral::Neutral(std::string name, unsigned int parentId):PyMooseBase(className, name, parentId)
{
}

Neutral::Neutral(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent)
{
}

Neutral::~Neutral()
{
}

const std::string& Neutral::getType()
{
   return className;
}
#endif
