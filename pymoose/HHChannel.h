#ifndef _pymoose_HHChannel_h
#define _pymoose_HHChannel_h
#include "PyMooseBase.h"
class HHChannel : public PyMooseBase
{    public:
        static const std::string className;
        HHChannel(Id id);
        HHChannel(std::string path);
        HHChannel(std::string name, Id parentId);
        HHChannel(std::string name, PyMooseBase* parent);
        ~HHChannel();
        const std::string& getType();
        double __get_Gbar() const;
        void __set_Gbar(double Gbar);
        double __get_Ek() const;
        void __set_Ek(double Ek);
        double __get_Xpower() const;
        void __set_Xpower(double Xpower);
        double __get_Ypower() const;
        void __set_Ypower(double Ypower);
        double __get_Zpower() const;
        void __set_Zpower(double Zpower);
        int __get_instant() const;
        void __set_instant(int instant);
        double __get_Gk() const;
        void __set_Gk(double Gk);
        double __get_Ik() const;
        void __set_Ik(double Ik);
        int __get_useConcentration() const;
        void __set_useConcentration(int useConcentration);
        double __get_IkSrc() const;
        void __set_IkSrc(double IkSrc);
        double __get_concen() const;
        void __set_concen(double concen);
    
    void createTable(std::string gate, unsigned int divs, double min, double max);    
    void tweakAlpha(std::string gate);
    void tweakTau(std::string gate);
    void setupAlpha(std::string gate, vector <double> params);
    void setupTau(std::string gate, vector <double> params);
    
};
#endif
