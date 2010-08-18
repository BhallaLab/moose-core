#ifndef _pymoose_glcell_h
#define _pymoose_glcell_h

#include "PyMooseBase.h"

#include "Neutral.h"
namespace pymoose{

	class GLcell : public Neutral
	{
	public:
		static const std::string className_;
		GLcell( Id id );
		GLcell( std::string path );
		GLcell( std::string name, Id parentId );
		GLcell( std::string name, PyMooseBase& parent );
		GLcell( const GLcell& src, std::string name, PyMooseBase& parent );
		GLcell( const GLcell& src, std::string name, Id& parent );
		GLcell( const Id& src, std::string path );
		GLcell( const GLcell& src, std::string path );
		GLcell( const Id& src, std::string name, Id& parent );
		~GLcell();
                const std::string& getType();
		const std::string __get_vizpath() const;
		void __set_vizpath( const std::string vizpath );
		const std::string& __get_clientHost() const;
		void __set_clientHost( const std::string strClientHost );
		const std::string& __get_clientPort() const;
		void __set_clientPort( const std::string strClientPort );
		const std::string& __get_attributeName() const;
		void __set_attributeName( const std::string strAttributeName );
		double __get_changeThreshold() const;
		void __set_changeThreshold( double changeThreshold );
		double __get_VScale() const;
		void __set_VScale( double vScale );
		const std::string& __get_syncMode() const;
		void __set_syncMode( const std::string& strSyncMode );
		const std::string& __get_bgColor() const;
		void __set_bgColor( const std::string strBgColor );
		double __get_highValue() const;
		void __set_highValue( double highValue );
		double __get_lowValue() const;
		void __set_lowValue( double lowValue );
	};
}

#endif
