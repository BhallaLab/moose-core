#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import socket
import time
import tarfile
import tempfile

def gen_prefix( msg, maxlength = 10 ):
    msg = '>%s' % msg
    if len(msg) < maxlength:
        msg += ' ' * (maxlength - len(msg))
    return msg[:maxlength].encode( 'utf-8' )

def gen_payload( args ):
    path = args.path
    archive = os.path.join(tempfile.mkdtemp(), 'data.tar.bz2')

    # This mode (w|bz2) is suitable for streaming. The blocksize is default to
    # 20*512 bytes. We change this to 2048
    with tarfile.open(archive, 'w|bz2', bufsize=2048 ) as h:
        if len(args.main) > 0:
            for i, f in enumerate(args.main):
                h.add(f, arcname=os.path.join(os.path.dirname(f),'__main__%d.py'%i))
        h.add(path)

    with open(archive, 'rb') as f:
        data = f.read()
    return data

def offload( args ):
    zfile = create_zipfile( args.path )
    send_zip( zfile )

def loop( sock ):
    sock.settimeout(1e-2)
    while True:
        try:
            d = sock.recv(10).strip()
            if len(d) > 0:
                print(d)
        except socket.timeout as e:
            print( '.', end='' )
            sys.stdout.flush()

def main( args ):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        host, port = args.server.split(':')
        sock.connect( (host, int(port)) )
        sock.settimeout(1)
    except Exception as e:
        print( "[ERROR] Failed to connect to %s... " % args.server )
        print( e )
        quit()

    data = gen_payload( args )
    data = b'<TARFILE>' + data + b'</TARFILE>'
    print( "[INFO ] Total data to send : %d bytes " % len(data), end = '')
    sock.sendall( data )
    print( '   [SENT]' )
    time.sleep(1)
    print( sock.recv(100) )
    sock.close()

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Submit a job to moose server.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path', metavar='path'
        , help = 'File or directory to execute on server.'
        )
    parser.add_argument('--main', '-m', nargs = '+'
        , required = False, default = []
        , help = 'In case of multiple files, scripts to execute'
                ' on the server, e.g. -m file1.py -m file2.py.'
                ' If not given, server will try to guess the best option.'
        )
    parser.add_argument('--server', '-s'
        , required = False, type=str, default='localhost:31417'
        , help = 'IP address and PORT number of moose server e.g.'
                 ' 172.16.1.2:31416'
        )
    class Args: pass
    args = Args()
    parser.parse_args(namespace=args)
    main(args)
