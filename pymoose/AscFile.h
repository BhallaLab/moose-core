#ifndef _pymoose_AscFile_h
#define _pymoose_AscFile_h
#include "PyMooseBase.h"
namespace pymoose{
    class AscFile : public PyMooseBase
    {      public:
        static const std::string className_;
        AscFile(Id id);
        AscFile(std::string path);
        AscFile(std::string name, Id parentId);
        AscFile(std::string name, PyMooseBase& parent);
        AscFile( const AscFile& src, std::string name, PyMooseBase& parent);
        AscFile( const AscFile& src, std::string name, Id& parent);
        AscFile( const AscFile& src, std::string path);
        AscFile( const Id& src, std::string name, Id& parent);
        ~AscFile();
        const std::string& getType();
            string __get_fileName() const;
            void __set_fileName(string fileName);
            int __get_appendFlag() const;
            void __set_appendFlag(int appendFlag);
    };
}
#endif
