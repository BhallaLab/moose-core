#!/usr/bin/env python

"""db.py: This module is only for debugging purpose.

Last modified: Wed Dec 25, 2013  08:00PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import sqlite3 as sql 


class DebugDB:

    def __init__(self, path):
        self.dbpath = path
        self.db = sql.connect(self.dbpath)
        self.c = self.db.cursor()

    def initDB(self):
        query = '''CREATE TABLE IF NOT EXISTS paths (
                path VARCHAR NOT NULL
                , type VARCHAR NOT NULL
                , comment TEXT
                , PRIMARY KEY path
                '''
        self.c.execute(query)
        self.commit()

    def commit(self):
        self.db.commit()

    def insertPath(self, path, type='', comment='No comment'):
        query = '''INSERT INTO paths (path, type, comment) VALUES (?, ?, ?)'''
        self.c.execute(query, (path, type, comment))

    def insertPaths(self, paths):
        [self.insertPath(p) for p in paths]
        self.commit()
