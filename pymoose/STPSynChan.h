#ifndef _pymoose_STPSynChan_h
#define _pymoose_STPSynChan_h
#include "SynChan.h"
namespace pymoose{
class PyMooseBase;
class SynChan;
class STPSynChan : public SynChan
{
  public:
    static const std::string className_;
    STPSynChan(std::string className, std::string objectName, Id parentId);
    STPSynChan(std::string className, std::string path);
    STPSynChan(std::string className, std::string objectName, PyMooseBase& parent);
    STPSynChan(Id id);
    STPSynChan(std::string path);
    STPSynChan(std::string name, Id parentId);
    STPSynChan(std::string name, PyMooseBase& parent);
    STPSynChan( const STPSynChan& src, std::string name, PyMooseBase& parent);
    STPSynChan( const STPSynChan& src, std::string name, Id& parent);
    STPSynChan( const STPSynChan& src, std::string path);
    STPSynChan( const Id& src, std::string name, Id& parent);
    STPSynChan( const Id& src, std::string path);
    ~STPSynChan();
    const std::string& getType();
    double __get_tauD1() const;
    void __set_tauD1(double tauD1);
    double __get_tauD2() const;
    void __set_tauD2(double tauD2);
    double __get_tauF() const;
    void __set_tauF(double tauF);
    double __get_deltaF() const;
    void __set_deltaF(double deltaF);
    double __get_d1() const;
    void __set_d1(double d1);
    double __get_d2() const;
    void __set_d2(double d2);
    double getF(const unsigned int& index);

    double getD1(const unsigned int& index);
        
    double getD2(const unsigned int& index);

    double getInitD1(const unsigned int& index);

    double getInitD2(const unsigned int& index);

    double getInitF(const unsigned int& index);

    double getInitPr(const unsigned int& index);

    double getPr(const unsigned int& index);

    void setInitPr(const unsigned int& index, double value);

    void setInitF(const unsigned int& index, double value);

    void setInitD1(const unsigned int& index, double value);

    void setInitD2(const unsigned int& index, double value);

};
}
#endif
