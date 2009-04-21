#ifndef _pymoose_HHChannel_h
#define _pymoose_HHChannel_h
#include "PyMooseBase.h"
namespace pymoose{
    class HHChannel : public PyMooseBase
    {      public:
        static const std::string className_;
        HHChannel(Id id);
        HHChannel(std::string path);
        HHChannel(std::string name, Id parentId);
        HHChannel(std::string name, PyMooseBase& parent);
        HHChannel( const HHChannel& src, std::string name, PyMooseBase& parent);
        HHChannel( const HHChannel& src, std::string name, Id& parent);
        HHChannel( const HHChannel& src, std::string path);
        HHChannel( const Id& src, std::string name, Id& parent);
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
            double __get_X() const;
            void __set_X(double X);
            double __get_Y() const;
            void __set_Y(double Y);
            double __get_Z() const;
            void __set_Z(double Z);
            double __get_initX() const;
            void __set_initX(double initX);
            double __get_initY() const;
            void __set_initY(double initY);
            double __get_initZ() const;
            void __set_initZ(double initZ);
            int __get_useConcentration() const;
            void __set_useConcentration(int useConcentration);
    };
}
#endif
