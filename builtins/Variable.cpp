// Variable.cpp ---
//
// Filename: Variable.cpp
// Description:
// Author: Subhasis Ray
// Maintainer: Dilawar Singh <dilawars@ncbs.res.in>
// Created: Fri May 30 19:56:06 2014 (+0530)

#include "header.h"
#include "Variable.h"
#include "Function.h"

const Cinfo * Variable::initCinfo()
{
    static ValueFinfo< Variable, double > value(
        "value",
        "Variable value",
        &Variable::setValue,
        &Variable::getValue
        );

    static DestFinfo input(
        "input",
        "Handles incoming variable value.",
        new EpFunc1< Variable, double >( &Variable::epSetValue )
        );

    static Finfo * variableFinfos[] = {
        &value,
        &input
    };

    static string doc[] = {
        "Name", "Variable",
        "Author", "Subhasis Ray",
        "Description", "Variable for storing double values. This is used in Function class."
    };

    static Dinfo< Variable > dinfo;

    static Cinfo variableCinfo("Variable",
            Neutral::initCinfo(),
            variableFinfos,
            sizeof(variableFinfos) / sizeof(Finfo*),
            &dinfo,
            doc,
            sizeof(doc) / sizeof(string),
            true // is FieldElement, not to be created directly
            );
    return & variableCinfo;
}

static const Cinfo * variableCinfo = Variable::initCinfo();

// This imitates Synapse::addMsgCallback
// but does not seem to be used anywhere
// - Subha, Tue Sep  9 19:37:11 IST 2014
void Variable::addMsgCallback(const Eref& e, const string& finfoName, ObjId msg, unsigned int msgLookup)
{
}
