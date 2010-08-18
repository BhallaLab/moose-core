#ifndef _pymoose_HHGate_h
#define _pymoose_HHGate_h
#include "PyMooseBase.h"
#include "Neutral.h"
#include "Interpol.h"
namespace pymoose{

    class HHGate : public Neutral
    {      public:
        static const std::string className_;
        HHGate(std::string className, std::string name, Id parentId);
        HHGate(std::string className, std::string path);
        HHGate(std::string className, std::string objectName, PyMooseBase& parent);
        HHGate(Id id);
        HHGate(std::string path);
        HHGate(std::string name, Id parentId);
        HHGate(std::string name, PyMooseBase& parent);
        HHGate( const HHGate& src, std::string name, PyMooseBase& parent);
        HHGate( const HHGate& src, std::string name, Id& parent);
        HHGate( const HHGate& src, std::string path);
        HHGate( const Id& src, std::string name, Id& parent);
        HHGate( const Id& src, std::string path);
        ~HHGate();
        const std::string& getType();
        // These are manually inserted
        Interpol* __get_A() const;
        Interpol* __get_B() const;
        void tabFill(int xdivs, int mode);
        void setupAlpha(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size=3000, double min=-0.1, double max=0.05);
        void setupTau(double AA, double AB, double AC , double AD, double AF, double BA, double BB, double BC, double BD, double BF, double size=3000, double min=-0.1, double max=0.05);
        void tweakAlpha();
        void tweakTau();
        
    };
}

#endif
