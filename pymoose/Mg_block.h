#ifndef _pymoose_Mg_block_h
#define _pymoose_Mg_block_h
#include "PyMooseBase.h"
#include "Neutral.h"
namespace pymoose{

    class Mg_block : public Neutral
    {      public:
        static const std::string className_;
        Mg_block(Id id);
        Mg_block(std::string path);
        Mg_block(std::string name, Id parentId);
        Mg_block(std::string name, PyMooseBase& parent);
        Mg_block( const Mg_block& src, std::string name, PyMooseBase& parent);
        Mg_block( const Mg_block& src, std::string name, Id& parent);
        Mg_block( const Mg_block& src, std::string path);
        Mg_block( const Id& src, std::string name, Id& parent);
        Mg_block( const Id& src, std::string path);
        ~Mg_block();
        const std::string& getType();
        double __get_KMg_A() const;
        void __set_KMg_A(double KMg_A);
        double __get_KMg_B() const;
        void __set_KMg_B(double KMg_B);
        double __get_CMg() const;
        void __set_CMg(double CMg);
        double __get_Ik() const;
        void __set_Ik(double Ik);
        double __get_Gk() const;
        void __set_Gk(double Gk);
        double __get_Ek() const;
        void __set_Ek(double Ek);
        double __get_Zk() const;
        void __set_Zk(double Zk);
    };
}

#endif
