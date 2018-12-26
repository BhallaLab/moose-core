/***
 *    Description:  Stream table data to a socket.
 */

#include <algorithm>
#include <sstream>
#include <unistd.h>

#include "../basecode/global.h"
#include "../basecode/header.h"
#include "../scheduling/Clock.h"
#include "../utility/utility.h"
#include "../shell/Shell.h"
#include "SocketStreamer.h"

const Cinfo* SocketStreamer::initCinfo()
{
    /*-----------------------------------------------------------------------------
     * Finfos
     *-----------------------------------------------------------------------------*/
    static ValueFinfo< SocketStreamer, size_t > port(
        "port"
        , "Set port number for streaming. This Streamer will send data every second."
        , &SocketStreamer::setPort
        , &SocketStreamer::getPort
    );

    static ReadOnlyValueFinfo< SocketStreamer, size_t > numTables (
        "numTables"
        , "Number of Tables handled by SocketStreamer "
        , &SocketStreamer::getNumTables
    );

    /*-----------------------------------------------------------------------------
     *
     *-----------------------------------------------------------------------------*/
    static DestFinfo process(
        "process"
        , "Handle process call"
        , new ProcOpFunc<SocketStreamer>(&SocketStreamer::process)
    );

    static DestFinfo reinit(
        "reinit"
        , "Handles reinit call"
        , new ProcOpFunc<SocketStreamer> (&SocketStreamer::reinit)
    );

    static DestFinfo addTable(
        "addTable"
        , "Add a table to SocketStreamer"
        , new OpFunc1<SocketStreamer, Id>(&SocketStreamer::addTable)
    );

    static DestFinfo addTables(
        "addTables"
        , "Add many tables to SocketStreamer"
        , new OpFunc1<SocketStreamer, vector<Id> >(&SocketStreamer::addTables)
    );

    static DestFinfo removeTable(
        "removeTable"
        , "Remove a table from SocketStreamer"
        , new OpFunc1<SocketStreamer, Id>(&SocketStreamer::removeTable)
    );

    static DestFinfo removeTables(
        "removeTables"
        , "Remove tables -- if found -- from SocketStreamer"
        , new OpFunc1<SocketStreamer, vector<Id>>(&SocketStreamer::removeTables)
    );

    /*-----------------------------------------------------------------------------
     *  ShareMsg definitions.
     *-----------------------------------------------------------------------------*/
    static Finfo* procShared[] =
    {
        &process, &reinit, &addTable, &addTables, &removeTable, &removeTables
    };

    static SharedFinfo proc(
        "proc",
        "Shared message for process and reinit",
        procShared, sizeof( procShared ) / sizeof( const Finfo* )
    );

    static Finfo * socketStreamFinfo[] =
    {
        &port, &proc, &numTables
    };

    static string doc[] =
    {
        "Name", "SocketStreamer",
        "Author", "Dilawar Singh (@dilawar, github),2018",
        "Description", "SocketStreamer: Stream moose.Table data to a socket.\n"
    };

    static Dinfo< SocketStreamer > dinfo;

    static Cinfo tableStreamCinfo(
        "SocketStreamer",
        TableBase::initCinfo(),
        socketStreamFinfo,
        sizeof( socketStreamFinfo )/sizeof(Finfo *),
        &dinfo,
        doc,
        sizeof(doc) / sizeof(string)
    );

    return &tableStreamCinfo;
}

static const Cinfo* tableStreamCinfo = SocketStreamer::initCinfo();

// Class function definitions

SocketStreamer::SocketStreamer() :
    format_("csv")
    , alreadyStreaming_(false)
    , sockfd_(-1)
    , clientfd_(-1)
    , ip_( TCP_SOCKET_IP )
    , port_( TCP_SOCKET_PORT )
    , numMaxClients_(1)
{
    // Not all compilers allow initialization during the declaration of class
    // methods.
    columns_.push_back( "time" );               /* First column is time. */
    tables_.resize(0);
    tableIds_.resize(0);
    tableTick_.resize(0);
    tableDt_.resize(0);

    // This should only be called once. 
    initServer();

    // Launch a thread in background which monitors the any client trying to
    // make connection to server.
    auto t = std::thread(&SocketStreamer::connect, this);
    t.detach();
    tm_["connect"] = std::move(t);
}

SocketStreamer& SocketStreamer::operator=( const SocketStreamer& st )
{
    return *this;
}


SocketStreamer::~SocketStreamer()
{
    // Now cleanup the socket as well.
    if(sockfd_ > 0)
    {
        LOG(moose::debug, "Closing socket " << sockfd_ );
        shutdown(sockfd_, SHUT_RD);
        close(sockfd_);
    }

    // Remember we created a background process to monitor the client. Terminate
    // it now. May be a good idea to wait for a little bit.
    stopThread( "connect" );
    sleep(0.01);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Stop a thread. 
 * See: http://www.bo-yang.net/2017/11/19/cpp-kill-detached-thread
 *
 * @Param tname name of thread.
 */
/* ----------------------------------------------------------------------------*/
void SocketStreamer::stopThread(const std::string& tname)
{
    ThreadMap::const_iterator it = tm_.find(tname);
    if (it != tm_.end()) 
    {
        it->second.std::thread::~thread(); // thread not killed
        tm_.erase(tname);
        LOG(moose::debug, "Thread " << tname << " killed." );
    }
}


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Connect to a client. If already connected to one, then do not make
 * any more connections.
 */
/* ----------------------------------------------------------------------------*/
void SocketStreamer::listenToClients(size_t numMaxClients)
{
    assert(0 < sockfd_ );
    assert( numMaxClients > 0 );
    numMaxClients_ = numMaxClients;
    if(-1 == listen(sockfd_, numMaxClients_))
    {
        LOG(moose::error, "Failed listen()" << strerror(errno) );
    }
}

void SocketStreamer::initServer( void )
{
    // Create a blocking socket.
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    int on = 1;

    // One can set socket option using setsockopt function. See manual page
    // for details. We are making it 'reusable'.
    if(0 > setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, (const char *)&on, sizeof(on)))
    {
        LOG(moose::warning, "Warn: setsockopt() failed");
        return;
    }

#ifdef SO_REUSEPORT
    if(0 > setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, (const char *)&on, sizeof(on)))
    {
        LOG(moose::warning, "Warn: setsockopt() failed");
        return;
    }
#endif

    addr_.sin_family = AF_INET;
    addr_.sin_addr.s_addr = INADDR_ANY;
    addr_.sin_port = htons( port_ );

    // Bind.
    if(0 > bind(sockfd_, (struct sockaddr*) &addr_, sizeof(addr_)))
    {
        LOG(moose::warning, "Warn: Failed to create server at " << ip_ << ":" << port_
            << ". File descriptor: " << sockfd_
            << ". Erorr: " << strerror(errno)
           );
        return;
    }
    else
        LOG(moose::debug, "Successfully bound socket." );

    LOG(moose::debug,  "Successfully created SocketStreamer server: " << sockfd_);

    //  Listen for incoming clients. This function does nothing if connection is
    //  already made.
    listenToClients(1);
}


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Stream data over socket.
 *
 * @Returns True of success, false otherwise. It is callee job to clean up data_
 *          on a successful return from this function.
 */
/* ----------------------------------------------------------------------------*/
bool SocketStreamer::streamData( )
{
    if( clientfd_ > 0)
    {
        buffer_ += dataToString();
        int sendBytes = send(clientfd_, buffer_.c_str(), buffer_.size(), MSG_MORE);
        if(0 > sendBytes)
        {
            LOG(moose::warning, "Failed to send. Error: " << strerror(errno)
                << ". client id: " << clientfd_ );
            return false;
        }
        // Send sendbytes has been sent. Remove as many characters from the msg
        // and append to buffer.
        // cout << "Send bytes " << sendBytes << " " << buffer_ << endl;
        buffer_ = buffer_.erase(0, sendBytes);

        // clear up the tables.
        for( auto t : tables_ )
            t->clearVec();

        return true;
    }
    else
        LOG(moose::debug, "No client found to stream data. ClientFD: " << clientfd_ );
    return false;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Convert table to string (use scientific notation). 
 *
 * @Returns String in JSON like format.
 */
/* ----------------------------------------------------------------------------*/
string SocketStreamer::dataToString( )
{
    stringstream ss;
    ss.precision( 7 );
    ss << std::scientific;

    // Else stream the data.
    ss << "{";
    for( size_t i = 0; i < tables_.size(); i++)
    {
        ss << "\"" << columns_[i+1] << "\":[";

        auto v = tables_[i]->data();

        // CSV.
        for( size_t ii = 0; ii < v.size()-1; ii++)
            ss << v[ii] << ',';
        ss << v.back();

        // Remove the last ,
        ss << "],";

    }

    // csv: remove last ,
    string res = ss.str();

    if( ',' == res.back())
        res.pop_back();
    res += "}";
    return res;
}

void SocketStreamer::connect( )
{
    Clock* clk = reinterpret_cast<Clock*>( Id(1).eref().data() );
    clientfd_ = accept(sockfd_, NULL, NULL);
    LOG(moose::debug, "Client " << clientfd_ << " is connected." );
}

/**
 * @brief Reinit. We make sure it is non-blocking.
 *
 * @param e
 * @param p
 */
void SocketStreamer::reinit(const Eref& e, ProcPtr p)
{

    // If no incoming connection found. Disable it.
    if( tables_.size() == 0 )
    {
        moose::showWarn( "No table found. Disabling SocketStreamer.\nDid you forget" 
                " to call addTables() on SocketStreamer object."
                );
        e.element()->setTick( -2 );             /* Disable process */
        return;
    }

    Clock* clk = reinterpret_cast<Clock*>( Id(1).eref().data() );

    // Push each table dt_ into vector of dt
    for( size_t i = 0; i < tables_.size(); i++)
    {
        Id tId = tableIds_[i];
        int tickNum = tId.element()->getTick();
        tableDt_.push_back( clk->getTickDt( tickNum ) );
    }
}

/**
 * @brief This function is called at its clock tick.
 *
 * @param e
 * @param p
 */
void SocketStreamer::process(const Eref& e, ProcPtr p)
{
    // cout << "Calling process" << endl;
    streamData();
}

/**
 * @brief Add a table to streamer.
 *
 * @param table Id of table.
 */
void SocketStreamer::addTable( Id table )
{
    // If this table is not already in the vector, add it.
    for( size_t i = 0; i < tableIds_.size(); i++)
        if( table.path() == tableIds_[i].path() )
            return;                             /* Already added. */

    Table* t = reinterpret_cast<Table*>(table.eref().data());
    tableIds_.push_back( table );
    tables_.push_back( t );
    tableTick_.push_back( table.element()->getTick() );

    // NOTE: If user can make sure that names are unique in table, using name is
    // better than using the full path.
    if( t->getColumnName().size() > 0 )
        columns_.push_back( t->getColumnName( ) );
    else
        columns_.push_back( moose::moosePathToUserPath( table.path() ) );
}

/**
 * @brief Add multiple tables to SocketStreamer.
 *
 * @param tables
 */
void SocketStreamer::addTables( vector<Id> tables )
{
    if( tables.size() == 0 )
        return;
    for( vector<Id>::const_iterator it = tables.begin(); it != tables.end(); it++)
        addTable( *it );
}


/**
 * @brief Remove a table from SocketStreamer.
 *
 * @param table. Id of table.
 */
void SocketStreamer::removeTable( Id table )
{
    int matchIndex = -1;
    for (size_t i = 0; i < tableIds_.size(); i++)
        if( table.path() == tableIds_[i].path() )
        {
            matchIndex = i;
            break;
        }

    if( matchIndex > -1 )
    {
        tableIds_.erase( tableIds_.begin() + matchIndex );
        tables_.erase( tables_.begin() + matchIndex );
        columns_.erase( columns_.begin() + matchIndex );
    }
}

/**
 * @brief Remove multiple tables -- if found -- from SocketStreamer.
 *
 * @param tables
 */
void SocketStreamer::removeTables( vector<Id> tables )
{
    for( vector<Id>::const_iterator it = tables.begin(); it != tables.end(); it++)
        removeTable( *it );
}

/**
 * @brief Get the number of tables handled by SocketStreamer.
 *
 * @return  Number of tables.
 */
size_t SocketStreamer::getNumTables( void ) const
{
    return tables_.size();
}


void SocketStreamer::setPort( const size_t port )
{
    port_ = port;
}

size_t SocketStreamer::getPort( void ) const
{
    assert( port_ > 1 );
    return port_;
}

/* Set the format of all Tables */
void SocketStreamer::setFormat( string fmt )
{
    format_ = fmt;
}

/*  Get the format of all tables. */
string SocketStreamer::getFormat( void ) const
{
    return format_;
}
