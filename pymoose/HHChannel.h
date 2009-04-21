#ifndef _pymoose_HHChannel_h
#define _pymoose_HHChannel_h
#include "PyMooseBase.h"
namespace pymoose
{
    class HHChannel : public PyMooseBase
    {
      public:
        static const std::string className_;
        HHChannel(Id id);
        HHChannel(std::string path);
        HHChannel(std::string name, Id parentId);
        HHChannel(std::string name, PyMooseBase& parent);
        HHChannel(const HHChannel& src,std::string name, PyMooseBase& parent);
        HHChannel(const HHChannel& src,std::string name, Id& parent);
        HHChannel(const Id& src,std::string name, Id& parent);
        HHChannel(const HHChannel& src,std::string path);
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
        double __get_X() const;
        void __set_X(double X);
        double __get_Y() const;
        void __set_Y(double Y);
        double __get_Z() const;
        void __set_Z(double Z);
        double __get_initX() const;
        void __set_initX(double X);
        double __get_initY() const;
        void __set_initY(double Y);
        double __get_initZ() const;
        void __set_initZ(double Z);
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
        void setupAlpha(std::string gate, double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size = 3000, double min = -0.1, double max = 0.05);
        void setupTau(std::string gate, vector <double> params);
        void setupTau(std::string gate, double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size = 3000, double min = -0.1, double max=0.05);    
    };
}

#endif
