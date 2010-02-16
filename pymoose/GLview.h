#ifndef _pymoose_glview_h
#define _pymoose_glview_h

#include "PyMooseBase.h"

namespace pymoose
{
	class GLview : public PyMooseBase
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
		GLview( const Id& src, std::string name, Id& parent );
		~GLview();
		const std::string& getType();
		const std::string& __get_vizpath() const;
		void __set_vizpath( const std::string& vizpath );
		const std::string& __get_clientHost() const;
		void __set_clientHost( const std::string& clientHost );
		const std::string& __get_clientPort() const;
		void __set_clientPort( const std::string& clientPort );
		const std::string& __get_relPath() const;
		void __set_relPath( const std::string& relPath );
		const std::string& __get_value1Field() const;
		void __set_value1Field( const std::string& value1Field );
		double __get_value1Min() const;
		void __set_value1Min( double value1Min );
		double __get_value1Max() const;
		void __set_value1Max( double value1Max );
		const std::string& __get_value2Field() const;
		void __set_value2Field( const std::string& value2Field );
		double __get_value2Min() const;
		void __set_value2Min( double value2Min );
		double __get_value2Max() const;
		void __set_value2Max( double value2Max );
		const std::string& __get_value3Field() const;
		void __set_value3Field( const std::string& value3Field );
		double __get_value3Min() const;
		void __set_value3Min( double value3Min );
		double __get_value3Max() const;
		void __set_value3Max( double value3Max );
		const std::string& __get_value4Field() const;
		void __set_value4Field( const std::string& value4Field );
		double __get_value4Min() const;
		void __set_value4Min( double value4Min );
		double __get_value4Max() const;
		void __set_value4Max( double value4Max );
		const std::string& __get_value5Field() const;
		void __set_value5Field( const std::string& value5Field );
		double __get_value5Min() const;
		void __set_value5Min( double value5Min );
		double __get_value5Max() const;
		void __set_value5Max( double value5Max );
		const std::string& __get_bgColor() const;
		void __set_bgColor( const std::string& bgColor );
		const std::string& __get_syncMode() const;
		void __set_syncMode( const std::string& syncMode );
		const std::string& __get_gridMode() const;
		void __set_gridMode( const std::string& gridMode );
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

