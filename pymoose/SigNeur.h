#ifndef _pymoose_SigNeur_h
#define _pymoose_SigNeur_h
#include "PyMooseBase.h"
namespace pymoose{
    class SigNeur : public PyMooseBase
    {      public:
        static const std::string className;
        SigNeur(Id id);
        SigNeur(std::string path);
        SigNeur(std::string name, Id parentId);
        SigNeur(std::string name, PyMooseBase& parent);
        SigNeur( const SigNeur& src, std::string name, PyMooseBase& parent);
        SigNeur( const SigNeur& src, std::string name, Id& parent);
        SigNeur( const SigNeur& src, std::string path);
        SigNeur( const Id& src, std::string name, Id& parent);
        ~SigNeur();
        const std::string& getType();
            Id& __get_cellProto() const;
            void __set_cellProto(Id& cellProto);
            Id& __get_spineProto() const;
            void __set_spineProto(Id& spineProto);
            Id& __get_dendProto() const;
            void __set_dendProto(Id& dendProto);
            Id& __get_somaProto() const;
            void __set_somaProto(Id& somaProto);
            Id& __get_cell() const;
            Id& __get_spine() const;
            Id& __get_dend() const;
            Id& __get_soma() const;
            string __get_cellMethod() const;
            void __set_cellMethod(string cellMethod);
            string __get_spineMethod() const;
            void __set_spineMethod(string spineMethod);
            string __get_dendMethod() const;
            void __set_dendMethod(string dendMethod);
            string __get_somaMethod() const;
            void __set_somaMethod(string somaMethod);
            double __get_sigDt() const;
            void __set_sigDt(double sigDt);
            double __get_cellDt() const;
            void __set_cellDt(double cellDt);
            double __get_Dscale() const;
            void __set_Dscale(double Dscale);
            double __get_lambda() const;
            void __set_lambda(double lambda);
            int __get_parallelMode() const;
            void __set_parallelMode(int parallelMode);
            double __get_updateStep() const;
            void __set_updateStep(double updateStep);
            double __get_calciumScale() const;
            void __set_calciumScale(double calciumScale);
            string __get_dendInclude() const;
            void __set_dendInclude(string dendInclude);
            string __get_dendExclude() const;
            void __set_dendExclude(string dendExclude);
    };
}
#endif
