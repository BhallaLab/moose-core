/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _CINFO_H
#define _CINFO_H

//TODO: mpp needs to be changed to automatically preface std:: to
// types in std namespace.
#include <map>
#include <string>
#include <vector>
#include "RecvFunc.h"

class Conn;
class Element;
class Field;
class Finfo;

class Cinfo {
	friend class CinfoWrapper;
	public:
		Cinfo(const std::string& name,
			const std::string& author,
			const std::string& description,
			const std::string& baseName,
			Finfo** fieldArray,
			const unsigned long nFields,
			Element* (*createWrapper)(const std::string&, Element*,
				const Element*)
			);

		~Cinfo() {
				;
		}

		const std::string& name() const {
				return name_;
		}


		static const Cinfo* find(const std::string& name);

		// Element* create(const std::string& name) const;
		Element* create(const std::string& name,
			Element* parent, const Element* proto = 0) const;

		Field field( const std::string& name ) const;
		void listFields( std::vector< Finfo* >& ) const;

		// Locates fields having matching conn and recvFunc
		const Finfo* findMsg( const Conn* conn, RecvFunc func ) const;

		// Locates fields having matching inConn
		// const Finfo* findMsg( const Conn* conn ) const;

		// Cleans out all entries pertaining to a given Conn prior to
		// deleting it. Returns true only if the Conn actually makes
		// contact.
		// bool funcDrop( Element* e, const Conn* conn ) const;

		// Locates fields holding matching connections and 
		// recvFuncs for calling  remote objects.
		Finfo* findRemoteMsg( Conn* c, RecvFunc func ) const;

		// Sets up the lookup std::map, creates cinfo classes on /classes,
		// assigns base names, and initializes field equivalence.
		static void initialize();

		// True if current class is derived from other.
		// Always true if 'other' is an Element.
		bool isA( const Cinfo* other ) const;

	private:
		// Note that this is implemented as a function to bypass issues
		// of sequence of static initialization.
		static std::map<std::string, Cinfo*>& lookup() {
			static std::map<std::string, Cinfo*> lookup_;
			return lookup_;
		}

		const std::string name_;
		std::string author_;
		std::string description_;
		std::string baseName_;
		Finfo** fieldArray_;
		const unsigned long nFields_;
		Element* (*createWrapper_)(
			const std::string& name, Element* parent, const Element* proto);

//		static std::vector<Cinfo*>& Cinfotab();
//		static std::vector<Cinfo*>& basetype();
		const Cinfo* base_;

};

#endif // _CINFO_H
