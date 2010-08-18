#ifndef _pymoose_AscFile_h
#define _pymoose_AscFile_h

#include "PyMooseBase.h"
#include "Neutral.h"

namespace pymoose{

    class AscFile : public Neutral
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
        AscFile( const Id& src, std::string path);
        ~AscFile();
        const std::string& getType();
            const string&  __get_filename() const;
            void __set_filename(string filename);
            int __get_append() const;
            void __set_append(int append);
            int __get_time() const;
            void __set_time(int time);
            int __get_header() const;
            void __set_header(int header);
            const string&  __get_comment() const;
            void __set_comment(string comment);
            const string&  __get_delimiter() const;
            void __set_delimiter(string delimiter);
    };
}
#endif
