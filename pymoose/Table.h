#ifndef _pymoose_Table_h
#define _pymoose_Table_h
#include "PyMooseBase.h"
#include "Interpol.h"
class Table : public InterpolationTable
{    public:
        static const std::string className;
        Table(Id id);
        Table(std::string path);
        Table(std::string name, Id parentId);
        Table(std::string name, PyMooseBase* parent);
        ~Table();
        const std::string& getType();
        double __get_input() const;
        void __set_input(double input);
        double __get_output() const;
        void __set_output(double output);
        int __get_step_mode() const;
        void __set_step_mode(int step_mode);
        int __get_stepmode() const;
        void __set_stepmode(int stepmode);
        double __get_stepsize() const;
        void __set_stepsize(double stepsize);
        double __get_threshold() const;
        void __set_threshold(double threshold);
    // todo: tackle these two functiosn properly
//        double __get_tableLookup() const;
//        void __set_tableLookup(double tableLookup);
        double __get_outputSrc() const;
        void __set_outputSrc(double outputSrc);
        double __get_msgInput() const;
        void __set_msgInput(double msgInput);
        double __get_sum() const;
        void __set_sum(double sum);
        double __get_prd() const;
        void __set_prd(double prd);
};
#endif
