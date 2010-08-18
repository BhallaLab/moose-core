#ifndef _pymoose_glview_h
#define _pymoose_glview_h

#include "PyMooseBase.h"

#include "Neutral.h"
namespace pymoose{

	class GLview : public Neutral
	{
	public:
		static const std::string className_;
		GLview( Id id );
		GLview( std::string path );
		GLview( std::string name, Id parentId );
		GLview( std::string name, PyMooseBase& parent );
		GLview( const GLview& src, std::string name, PyMooseBase& parent );
		GLview( const GLview& src, std::string name, Id& parent );
		GLview( const GLview& src, std::string path );
		GLview( const Id& src, std::string path );
		GLview( const Id& src, std::string name, Id& parent );
		~GLview();
		const std::string& getType();
		std::string __get_vizpath() const;
		void __set_vizpath( std::string vizpath );
                std::string __get_clientHost() const;
		void __set_clientHost( std::string clientHost );
		std::string __get_clientPort() const;
		void __set_clientPort( std::string clientPort );
		std::string __get_relPath() const;
		void __set_relPath( std::string relPath );
		std::string __get_value1Field() const;
		void __set_value1Field( std::string value1Field );
		double __get_value1Min() const;
		void __set_value1Min( double value1Min );
		double __get_value1Max() const;
		void __set_value1Max( double value1Max );
		std::string __get_value2Field() const;
		void __set_value2Field( std::string value2Field );
		double __get_value2Min() const;
		void __set_value2Min( double value2Min );
		double __get_value2Max() const;
		void __set_value2Max( double value2Max );
		std::string __get_value3Field() const;
		void __set_value3Field( std::string value3Field );
		double __get_value3Min() const;
		void __set_value3Min( double value3Min );
		double __get_value3Max() const;
		void __set_value3Max( double value3Max );
		std::string __get_value4Field() const;
		void __set_value4Field( std::string value4Field );
		double __get_value4Min() const;
		void __set_value4Min( double value4Min );
		double __get_value4Max() const;
		void __set_value4Max( double value4Max );
		std::string __get_value5Field() const;
		void __set_value5Field( std::string value5Field );
		double __get_value5Min() const;
		void __set_value5Min( double value5Min );
		double __get_value5Max() const;
		void __set_value5Max( double value5Max );
		std::string __get_bgColor() const;
		void __set_bgColor( std::string bgColor );
		std::string __get_syncMode() const;
		void __set_syncMode( std::string syncMode );
		std::string __get_gridMode() const;
		void __set_gridMode( std::string gridMode );
		unsigned int __get_colorVal() const;
		void __set_colorVal( unsigned int colorVal );
		unsigned int __get_morphVal() const;
		void __set_morphVal( unsigned int morphVal );
		unsigned int __get_xoffsetVal() const;
		void __set_xoffsetVal( unsigned int xoffsetVal );
		unsigned int __get_yoffsetVal() const;
		void __set_yoffsetVal( unsigned int yoffsetVal );
		unsigned int __get_zoffsetVal() const;
		void __set_zoffsetVal( unsigned int zoffsetVal );
	};
}

#endif

