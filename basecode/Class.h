/*******************************************************************
 * File:            Class.h
 * Description:      Wrapper for Cinfo which helps in introspcetion
 *                   and allows access to class variables.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-10-13 20:57:45
 ********************************************************************/

#ifndef _CLASS_H
#define _CLASS_H
#include "header.h"
#include "moose.h"

const unsigned int BAD_TICK = ~0;
const unsigned int BAD_STAGE = ~0;
/**
   Wrapper for Cinfo objects
 */
class Class
{
  public:
    Class(string name="Class");
    
    static const string getName(const Element* e);
    static void setName(const Conn* conn,string name);
    
    static const string getAuthor(const Element* e);
    static const string getDescription(const Element* e);
    
    static void setClock(const Conn* conn, string function, Id tickId);
    static unsigned int getTick(const Element* e);    
    static void setTick(const Conn* conn, unsigned int tick);
    static unsigned int getStage(const Element* e);    
    static void setStage(const Conn* conn, unsigned int stage);
    static vector < string > getFieldList(const Element* e); /// returns a list of field names
        
  private:
    Cinfo* classInfo_;
    
};

#endif
