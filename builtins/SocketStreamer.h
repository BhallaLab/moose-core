/***
 *    Stream table data to a TCP socket.
 */

#ifndef  SocketStreamer_INC
#define  SocketStreamer_INC

#define STRINGSTREAM_DOUBLE_PRECISION       10

#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>

#include "StreamerBase.h"
#include "Table.h"

// If cmake does not set it, use the default port.
#ifndef TCP_SOCKET_PORT
#define TCP_SOCKET_PORT  31415
#endif

#ifndef TCP_SOCKET_IP
#define TCP_SOCKET_IP  "127.0.0.1"
#endif

// Before send() can be used with c++.
#define _XOPEN_SOURCE_EXTENDED 1

// cmake should set include path.
#include <sys/socket.h>
#include <sys/poll.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <netinet/in.h>

using namespace std;

class SocketStreamer : public StreamerBase
{

public:
    SocketStreamer();
    ~SocketStreamer();

    SocketStreamer& operator=( const SocketStreamer& st );

    // Initialize server.
    void initServer( void );

    // Make connection to client
    void listenToClients(size_t numMaxClients);

    /* Cleaup before quitting */
    void cleanUp( void );

    string getFormat( void ) const;
    void setFormat( string format );

    size_t getPort( void ) const;
    void setPort( const size_t port );

    // Stream data.
    bool streamData();
    void connect();

    size_t getNumTables( void ) const;

    void addTable( Id table );
    void addTables( vector<Id> tables);

    void removeTable( Id table );
    void removeTables( vector<Id> table );

    string dataToString();

    void stopThread(const std::string& tname);

    /** Dest functions.
     * The process function called by scheduler on every tick
     */
    void process(const Eref& e, ProcPtr p);

    /**
     * The reinit function called by scheduler for the reset
     */
    void reinit(const Eref& e, ProcPtr p);

    static const Cinfo * initCinfo();

private:

    bool alreadyStreaming_;

    // dt_ and tick number of Table's clock
    vector<double> tableDt_;
    vector<unsigned int> tableTick_;
    double currTime_;

    // Used for adding or removing tables
    vector<Id> tableIds_;
    vector<Table*> tables_;
    vector<string> columns_;

    /* Socket related */
    int sockfd_;        // socket file descriptor.
    int clientfd_;      // client file descriptor
    string ip_;         // ip_ address of server.
    unsigned short port_;
    struct sockaddr_in addr_;
    int numMaxClients_;

    // std::thread processThread_;
    typedef std::map<std::string, std::thread> ThreadMap;
    ThreadMap tm_;
    string buffer_;
};

#endif   /* ----- #ifndef SocketStreamer_INC  ----- */
