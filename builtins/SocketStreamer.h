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
#include "MooseSocketInfo.h"
#include "Table.h"

// If cmake does not set it, use the default port.
#ifndef TCP_SOCKET_PORT
#define TCP_SOCKET_PORT  31416
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
#include <sys/un.h>
#include <netinet/in.h>

// MSG_MORE is not defined in OSX. So stupid!
#ifndef MSG_MORE
#define MSG_MORE 0
#endif

using namespace std;


class Clock;


class SocketStreamer : public StreamerBase
{

public:
    SocketStreamer();
    ~SocketStreamer();

    SocketStreamer& operator=( const SocketStreamer& st );

    string getFormat( void ) const;
    void setFormat( string format );

    /*-----------------------------------------------------------------------------
     *  Socket Server
     *-----------------------------------------------------------------------------*/
    // Initialize server.
    void initServer( void );
    void initTCPServer( void );
    void initUDSServer( void );

    /* common configuration options */
    void configureSocketServer( void );

    // Make connection to client
    void listenToClients(size_t numMaxClients);

    /* Cleaup before quitting */
    void cleanUp( void );

    string getAddress( void ) const;
    void setAddress( const string addr );

    size_t getPort( void ) const;
    void setPort( const size_t port );

    /*-----------------------------------------------------------------------------
     *  Streaming data.
     *-----------------------------------------------------------------------------*/
    bool enoughDataToStream(size_t minsize=10);
    bool streamData();
    void connectAndStream( );

    size_t getNumTables( void ) const;

    void addTable( ObjId table );
    void addTables( vector<ObjId> tables);

    void removeTable( ObjId table );
    void removeTables( vector<ObjId> table );

    string dataToString(size_t& start_from);

    // void stopThread(const std::string& tname);

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

    // dt_ and tick number of Table's clock
    vector<double> tableDt_;
    vector<unsigned int> tableTick_;
    double currTime_;

    // Used for adding or removing tables
    vector<Id> tableIds_;
    vector<Table*> tables_;
    vector<string> columns_;

    /* Socket related */
    int numMaxClients_;
    int sockfd_;                                      // socket file descriptor.
    int clientfd_;                                    // client file descriptor

    // address holdder for TCP and UDS sockets.
    struct sockaddr_in sockAddrTCP_;
    struct sockaddr_un sockAddrUDS_;

    /* For data handling */
    bool all_done_ = false;
    bool isValid_ = true;
    std::thread processThread_;
    string buffer_;
    double thisDt_;

    size_t frameSize_ = 2048;

    // We need clk_ pointer for handling
    Clock* clk_ = nullptr;

    // Socket Info
    MooseSocketInfo sockInfo_;     
};

#endif   /* ----- #ifndef SocketStreamer_INC  ----- */
