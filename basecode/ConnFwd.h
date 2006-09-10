#ifndef _ConnFwd_h
#define _ConnFwd_h

#include "ElementFwd.h"

class Conn;
template< Element* (* F)(const Conn*) >class UniConn;
class UniConn2;
template< class T >class SynConn;
class PlainMultiConn;
class MultiConn;
class RelayConn;
class ReturnConn;
class MultiReturnConn;

#endif
