# -*- coding: utf-8 -*-
from __future__ import print_function, division

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2019, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import re
import os 
import time
import shutil
import socket 
import signal
import tarfile 
import tempfile 
import threading 
import logging
import helper 

# setup environment variables for the streamer starts working.
os.environ['MOOSE_SOCKET_STREAMER_ADDRESS'] = 'ghevar.ncbs.res.in:31416'

# create a logger for this server.
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename='moose_server.log',
        filemode='a'
        )
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
_logger = logging.getLogger('')
_logger.addHandler(console)

__all__ = [ 'serve' ]

# Global variable to stop all running threads.
stop_all_ = False

# Signal handler.
def handler(signum, frame):
    global stop_all_
    _logger.info( "User terminated all processes." )
    stop_all_ = True

# Install a signal handler.
signal.signal( signal.SIGINT, handler)

def split_data( data ):
    prefixLenght = 10
    return data[:prefixLenght].strip(), data[prefixLenght:]

def send_msg(msg, conn):
    if not msg.strip():
        return False
    _logger.debug( msg )
    msg = '%s>>> %s' % (socket.gethostname(), msg)
    conn.sendall( b'%010d%s' % (len(msg), msg))

def run(cmd, conn, cwd=None):
    _logger.info( "Executing %s" % cmd )
    oldCWD = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    for line in helper.execute(cmd.split()):
        send_msg(line, conn)
    os.chdir(oldCWD)

def recv_input(conn, size=1024):
    # first 10 bytes always tell how much to read next. Make sure the submit job
    # script has it
    d = conn.recv(10, socket.MSG_WAITALL)
    while len(d) < 10:
        try:
            d = conn.recv(10, socket.MSG_WAITALL)
        except Exception:
            _logger.error( "Error in format. First 6 bytes are size of msg." )
            continue
    d, data = int(d), b''
    while len(data) < d:
        data += conn.recv(d-len(data), socket.MSG_WAITALL)
    return data

def writeTarfile( data ):
    tfile = os.path.join(tempfile.mkdtemp(), 'data.tar.bz2')
    with open(tfile, 'wb' ) as f:
        _logger.info( "Writing %d bytes to %s" % (len(data), tfile))
        f.write(data)
    # Sleep for some time so that file can be written to disk.
    time.sleep(0.2)
    if not tarfile.is_tarfile(tfile):
        _logger.warn( 'Not a valid tar file: %s' % tfile)
        return None
    return tfile

def suffixMatplotlibStmt( filename ):
    outfile = '%s.1.py' % filename
    with open(filename, 'r') as f:
        txt = f.read()

    matplotlibText = """
#  from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def saveall(prefix='results', figs=None):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        fig.savefig('%s.%d.png' %(prefix,i) )
    plt.close()

try:
    #  multipage("results.pdf")
    saveall()
except Exception as e:
    print( e )
    """
    with open(outfile, 'w' ) as f:
        f.write( txt )
        f.write( matplotlibText )
    return outfile

def run_file(filename, conn, cwd=None):
    filename = suffixMatplotlibStmt(filename)
    run( "%s %s" % (sys.executable, filename), conn, cwd)

def extract_files(tfile, to):
    userFiles = []
    with tarfile.open(tfile, 'r' ) as f:
        userFiles = f.getnames( )
        try:
            f.extractall( to )
        except Exception as e:
            _logger.warn( e)
    # now check if all files have been extracted properly
    for f in userFiles:
        if not os.path.exists(f):
            _logger.error( "File %s could not be extracted." % f )
    return userFiles

def prepareMatplotlib( cwd ):
    with open(os.path.join(cwd, 'matplotlibrc'), 'w') as f:
        f.write( 'interactive : True' )

def send_bz2(conn, data):
    data = b'%010d%s' % (len(data), data)
    conn.sendall(data)

def sendResults(tdir, conn, notTheseFiles):
    # Only send new files.
    resdir = tempfile.mkdtemp()
    resfile = os.path.join(resdir, 'results.tar.bz2')
    with tarfile.open( resfile, 'w|bz2') as tf:
        for d, sd, fs in os.walk(tdir):
            for f in fs:
                fpath = os.path.join(d,f)
                if fpath not in notTheseFiles:
                    _logger.info( "Adding file %s" % f )
                    tf.add(os.path.join(d, f), f)

    time.sleep(0.01)
    # now send the tar file back to client
    with open(resfile, 'rb' ) as f:
        data = f.read()
        _logger.info( 'Total bytes to send to client: %d' % len(data))
        send_bz2(conn, data)
    shutil.rmtree(resdir)

def find_files_to_run( files ):
    """Any file name starting with __main is to be run.
    Many such files can be recieved by client.
    """
    toRun = []
    for f in files:
        if '__main' in os.path.basename(f):
            toRun.append(f)
    if toRun:
        return toRun
    # Else guess.
    if len(files) == 1:
        return files

    for f in files:
        with open(f, 'r' ) as fh:
            txt = fh.read()
            if re.search(r'def\s+main\(', txt):
                if re.search('^\s+main\(\S+?\)', txt):
                    toRun.append(f)
    return toRun

def simulate( tfile, conn ):
    """Simulate a given tar file.
    """
    tdir = os.path.dirname( tfile )
    os.chdir( tdir )
    userFiles = extract_files(tfile, tdir)
    # Now simulate.
    toRun = find_files_to_run(userFiles)
    if len(toRun) < 1:
        return 1
    prepareMatplotlib(tdir)
    status, msg = 0, ''
    for _file in toRun:
        try:
            run_file(_file, conn, tdir) 
        except Exception as e:
            msg += str(e)
            status = 1
    return status, msg

def savePayload( conn ):
    data = recv_input(conn)
    tarfileName = writeTarfile(data)
    return tarfileName, len(data)

def handle_client(conn, ip, port):
    isActive = True
    while isActive:
        tarfileName, nBytes = savePayload(conn)
        if tarfileName is None:
            _logger.warn( "Could not recieve data." )
            isActive = False
        if not os.path.isfile(tarfileName):
            send_msg("[ERROR] %s is not a valid tarfile. Retry"%tarfileName, conn)
            break

        # list of files before the simulation.
        notthesefiles = helper.find_files(os.path.dirname(tarfileName))
        res, msg = simulate( tarfileName, conn )
        if not res:
            send_msg( "Failed to run simulation: %s" % msg, conn)
            isActive = False
            time.sleep(0.1)
        send_msg('>DONE SIMULATION', conn)
        # Send results after DONE is sent.
        sendResults(os.path.dirname(tarfileName), conn, notthesefiles)

def start_server( host, port, max_requests = 10 ):
    global stop_all_
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        soc.bind( (host, port))
        _logger.info( "Server created %s:%s" %(host,port) )
    except Exception as e:
        _logger.error( "Failed to bind: %s" % e)
        quit(1)

    # listen upto 100 of requests
    soc.listen(max_requests)
    while True:
        conn, (ip, port) = soc.accept()
        _logger.info( "Connected with %s:%s" % (ip, port) )
        try:
            t = threading.Thread(target=handle_client, args=(conn, ip, port)) 
            t.start()
        except Exception as e:
            _logger.warn(e)
        if stop_all_:
            break
    soc.close()

def serve(host, port):
    start_server(host, port)

def main( args ):
    global stop_all_
    host, port = args.host, args.port
    try:
        serve(host, port)
    except KeyboardInterrupt:
        stop_all_ = True
        quit(1)

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Run MOOSE server.'''
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument( '--help', action='help', help='Show this msg and exit')
    parser.add_argument('--host', '-h'
        , required = False, default = socket.gethostbyname(socket.gethostname())
        , help = 'Server Name'
        )
    parser.add_argument('--port', '-p'
        , required = False, default = 31417, type=int
        , help = 'Port number'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    try:
        main(args)
    except KeyboardInterrupt as e:
        stop_all_ = True
        quit(1)
