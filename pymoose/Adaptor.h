#ifndef _pymoose_Adaptor_h
#define _pymoose_Adaptor_h

#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose{
class Adaptor : public Neutral
{
  public:
    static const std::string className_;
    Adaptor(Id id);
    Adaptor(std::string path);
    Adaptor(std::string name, Id parentId);
    Adaptor(std::string name, PyMooseBase& parent);
    Adaptor( const Adaptor& src, std::string name, PyMooseBase& parent);
    Adaptor( const Adaptor& src, std::string name, Id& parent);
    Adaptor( const Adaptor& src, std::string path);
    Adaptor( const Id& src, std::string name, Id& parent);
    Adaptor( const Id& src, std::string path);
    ~Adaptor();
    const std::string& getType();
    double __get_inputOffset() const;
    void __set_inputOffset(double inputOffset);
    double __get_outputOffset() const;
    void __set_outputOffset(double outputOffset);
    double __get_scale() const;
    void __set_scale(double scale);
    double __get_output() const;
};

}
#endif
