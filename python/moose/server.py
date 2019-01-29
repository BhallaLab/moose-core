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
import tarfile 
import tempfile 
import threading 
import datetime
import subprocess

__all__ = [ 'serve' ]

stop_all_ = False

def split_data( data ):
    prefixLenght = 10
    return data[:prefixLenght].strip(), data[prefixLenght:]

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def send_msg(msg, conn):
    if not msg.strip():
        return False
    print( msg )
    msg = '%s >>> %s' % (socket.gethostname(), msg)
    conn.sendall(msg)

def run(cmd, conn, cwd=None):
    oldCWD = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    print( "[INFO ] Executing in %s" % os.getcwd() )
    for line in execute(cmd.split()):
        send_msg(line, conn)
    os.chdir(oldCWD)

def find_files_to_run( files ):
    """Any file name starting with __main is to be run."""
    toRun = []
    for f in files:
        if '__main' in os.path.basename(f):
            toRun.append(f)
    if toRun:
        return toRun

    # Then guess.
    if len(files) == 1:
        return files

    for f in files:
        with open(f, 'r' ) as fh:
            txt = fh.read()
            if re.search(r'def\s+main\(', txt):
                if re.search('^\s+main\(\S+?\)', txt):
                    toRun.append(f)
    return toRun

def recv_input(conn, size=1024):
    # first 10 bytes always tell how much to read next. Make sure the submit job
    # script has it
    d = conn.recv(10)
    while len(d) < 10:
        try:
            d = conn.recv(10)
        except Exception:
            print( "[ERROR] Error in format. First 6 bytes are size of msg." )
            continue
    data = conn.recv(int(d))
    return data

def writeTarfile( data ):
    tfile = os.path.join(tempfile.mkdtemp(), 'data.tar.bz2')
    with open(tfile, 'wb' ) as f:
        print( "[INFO ] Writing %d bytes to %s" % (len(data), tfile))
        f.write(data)
    # Sleep for some time so that file can be written to disk.
    time.sleep(0.1)
    if not tarfile.is_tarfile(tfile):
        send_msg("[ERROR] Not a valid tarfile %s. Please retry" % tfile)
        return None
    return tfile

def suffixMatplotlibStmt( filename ):
    outfile = '%s.1.py' % filename
    with open(filename, 'r') as f:
        txt = f.read()

    matplotlibText = '''
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

try:
    multipage("results.pdf")
    print( 'Saved all data to results.pdf' )
except Exception as e:
    pass
    '''
    with open(outfile, 'w' ) as f:
        f.write( txt )
        f.write( matplotlibText )
    return outfile

def run_file(filename, conn, cwd=None):
    print( '[INFO] Running %s' % filename )
    filename = suffixMatplotlibStmt(filename)
    run( "%s %s" % (sys.executable, filename), conn, cwd)
    print( '.... DONE' )

def extract_files(tfile, to):
    userFiles = []
    with tarfile.open(tfile, 'r' ) as f:
        userFiles = f.getnames( )
        try:
            f.extractall( to )
        except Exception as e:
            print( e)
    # now check if all files have been extracted properly
    success = True
    for f in userFiles:
        if not os.path.exists(f):
            print( "[ERROR] File %s could not be extracted." % f )
            success = False
    if success:
        os.remove(tfile)
    return userFiles

def prepareMatplotlib( cwd ):
    with open(os.path.join(cwd, 'matplotlibrc'), 'w') as f:
        f.write( 'interactive : True' )

def send_bz2(conn, data):
    data = b'%010d%s' % (len(data), data)
    conn.sendall(data)

def sendResults(tdir, conn, fromThisTime):
    # Only send new files.
    resdir = tempfile.mkdtemp()
    resfile = os.path.join(resdir, 'results.tar.bz2')

    with tarfile.open( resfile, 'w|bz2') as tf:
        for d, sd, fs in os.walk(tdir):
            for f in fs:
                if datetime.datetime.fromtimestamp(os.path.getmtime(f)) > fromThisTime:
                    print( "[INFO ] Adding file %s" % f )
                    tf.add(os.path.join(d, f))

    time.sleep(0.01)
    # now send the tar file back to client
    with open(resfile, 'rb' ) as f:
        data = f.read()
        print( 'Total bytes in result: %d' % len(data))
        send_bz2(conn, data)
    shutil.rmtree(resdir)

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
    [ run_file(_file, conn, tdir) for _file in toRun ]
    return userFiles

def savePayload( conn ):
    data = recv_input(conn)
    tarfileName = writeTarfile(data)
    return tarfileName

def handle_client(conn, ip, port):
    isActive = True
    while isActive:
        tarfileName = savePayload(conn)
        if tarfileName is None:
            isActive = False
        print( "[INFO ] PAYLOAD RECIEVED." )
        if not os.path.isfile(tarfileName):
            break
        startSimTime = datetime.datetime.now()
        isActive = False
        send_msg('>DONE SIMULATION', conn)
        # Send results after DONE is sent.
        sendResults(os.path.dirname(tarfileName), conn, startSimTime)

def start_server( host, port, max_requests = 10 ):
    global stop_all_
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        soc.bind( (host, port))
        print( "[INFO ] Server created %s:%s" %(host,port) )
    except Exception as e:
        print( "[ERROR] Failed to bind: %s" % str(sys.exec_info()))
        quit(1)

    # listen upto 100 of requests
    soc.listen(max_requests)
    while True:
        conn, (ip, port) = soc.accept()
        print( "[INFO ] Connected with %s:%s" % (ip, port) )
        try:
            t = threading.Thread(target=handle_client, args=(conn, ip, port)) 
            t.start()
        except Exception as e:
            print(e)
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
    main(args)
