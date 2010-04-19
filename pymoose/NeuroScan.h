#ifndef _pymoose_NeuroScan_h
#define _pymoose_NeuroScan_h
#include "PyMooseBase.h"
namespace pymoose{
    class NeuroScan : public PyMooseBase
    {      public:
        static const std::string className_;
        NeuroScan(Id id);
        NeuroScan(std::string path);
        NeuroScan(std::string name, Id parentId);
        NeuroScan(std::string name, PyMooseBase& parent);
        NeuroScan( const NeuroScan& src, std::string name, PyMooseBase& parent);
        NeuroScan( const NeuroScan& src, std::string name, Id& parent);
        NeuroScan( const NeuroScan& src, std::string path);
        NeuroScan( const Id& src, std::string name, Id& parent);
	NeuroScan( const Id& src, std::string path);
        ~NeuroScan();
        const std::string& getType();
            int __get_VDiv() const;
            void __set_VDiv(int VDiv);
            double __get_VMin() const;
            void __set_VMin(double VMin);
            double __get_VMax() const;
            void __set_VMax(double VMax);
            int __get_CaDiv() const;
            void __set_CaDiv(int CaDiv);
            double __get_CaMin() const;
            void __set_CaMin(double CaMin);
            double __get_CaMax() const;
            void __set_CaMax(double CaMax);
    };
}
#endif
