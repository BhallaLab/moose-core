# -*- python -*-
#
#       This was derived from shell.py [INRIA OpenAlea project] which
#       itself was modification of PyCute3.py of PyQwt project.
#

__doc__="""
This module is a QT4 version of PyCute3 python interpreter widget.
"""

import os, sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QTextEdit, QTextCursor
from PyQt4.QtCore import Qt

from mooseglobals import MooseGlobals

class MultipleRedirection:
    """ Dummy file which redirects stream to multiple file """

    def __init__(self, files):
        """ The stream is redirect to the file list 'files' """

        self.files = files

    def write(self, str):
        """ Emulate write function """

        for f in self.files:
            f.write(str)
            

class PyCute(QTextEdit):

    """
    PyCute is a Python shell for PyQt.

    Creating, displaying and controlling PyQt widgets from the Python command
    line interpreter is very hard, if not, impossible.  PyCute solves this
    problem by interfacing the Python interpreter to a PyQt widget.

    This class is inspired by PyCute.py : http://gerard.vermeulen.free.fr (GPL)
    """
    
    def __init__(self, interpreter, message="", log='', parent=None):
        """Constructor.
        @param interpreter : InteractiveInterpreter in which
        the code will be executed

        @param message : welcome message string
        
        @param 'log' : specifies the file in which the
        interpreter session is to be logged.

        @param  'parent' : specifies the parent widget.
        If no parent widget has been specified, it is possible to
        exit the interpreter by Ctrl-D.
        """

        QTextEdit.__init__(self, parent)
        self.setUndoRedoEnabled(True)
        self.interpreter = interpreter
        self.colorizer = SyntaxColor()

        # session log
        self.log = log or ''

        # to exit the main interpreter by a Ctrl-D if PyCute has no parent
        if parent is None:
            self.eofKey = Qt.Key_D
        else:
            self.eofKey = None

        # capture all interactive input/output 
        sys.stdout   = self
        sys.stderr   = MultipleRedirection((sys.stderr, self))
        sys.stdin    = self

        
        # last line + last incomplete lines
        self.line    = QtCore.QString()
        self.lines   = []
        # the cursor position in the last line
        self.point   = 0
        # flag: the interpreter needs more input to run the last lines. 
        self.more    = 0
        # flag: readline() is being used for e.g. raw_input() and input()
        self.reading = 0
        # history
        self.history = []
        self.pointer = 0
        self.cursor_pos   = 0

        # user interface setup
        self.setLineWrapMode(QTextEdit.NoWrap)
        
        # interpreter prompt.
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "

        # interpreter banner
        self.write('The shell running Python %s on %s.\n' %
                   (sys.version, sys.platform))
        self.write('Type "copyright", "credits" or "license"'
                   ' for more information on Python.\n')
        self.write(message+'\n\n')
        # self.write('This is the standard Shell.\n')
        self.write(sys.ps1)
        

    def get_interpreter(self):
        """ Return the interpreter object """

        return self.interpreter
        

    def moveCursor(self, operation, mode=QTextCursor.MoveAnchor):
        """
        Convenience function to move the cursor
        This function will be present in PyQT4.2
        """
        cursor = self.textCursor()
        cursor.movePosition(operation, mode)
        self.setTextCursor(cursor)
        

    def flush(self):
        """
        Simulate stdin, stdout, and stderr.
        """
        pass


    def isatty(self):
        """
        Simulate stdin, stdout, and stderr.
        """
        return 1
    

    def readline(self):
        """
        Simulate stdin, stdout, and stderr.
        """
        self.reading = 1
        self._clearLine()
        self.moveCursor(QTextCursor.End)
        while self.reading:
            QtGui.qApp.processEvents(QtCore.QEventLoop.WaitForMoreEvents)
        if self.line.length() == 0:
            return '\n'
        else:
            return str(self.line) 

    
    def write(self, text):
        """
        Simulate stdin, stdout, and stderr.
        """
        # The output of self.append(text) contains to many newline characters,
        # so work around QTextEdit's policy for handling newline characters.
        self.setUndoRedoEnabled(True)
        cursor = self.textCursor()

        cursor.movePosition(QTextCursor.End)

        pos1 = cursor.position()
        cursor.insertText(text)

        self.cursor_pos = cursor.position()
        self.setTextCursor(cursor)
        self.ensureCursorVisible ()

        # Set the format
        cursor.setPosition(pos1, QTextCursor.KeepAnchor)
        format = cursor.charFormat()
        format.setForeground( QtGui.QBrush(QtGui.QColor(0,0,0)))
        cursor.setCharFormat(format)

    def writelines(self, text):
        """
        Simulate stdin, stdout, and stderr.
        """
        map(self.write, text)


    def fakeUser(self, lines):
        """
        Simulate a user: lines is a sequence of strings, (Python statements).
        """
        for line in lines:
            self.line = QtCore.QString(line.rstrip())
            self.write(self.line)
            self.write('\n')
            self._run()

            
    def _run(self):
        """
        Append the last line to the history list, let the interpreter execute
        the last line(s), and clean up accounting for the interpreter results:
        (1) the interpreter succeeds
        (2) the interpreter fails, finds no errors and wants more line(s)
        (3) the interpreter fails, finds errors and writes them to sys.stderr
        """
        self.pointer = 0
        self.history.append(QtCore.QString(self.line))
        try:
            self.lines.append(str(self.line))
        except Exception,e:
            print e

        source = '\n'.join(self.lines)
        self.more = self.interpreter.runsource(source)

        if self.more:
            self.write(sys.ps2)
        else:
            self.write(sys.ps1)
            self.lines = []
        self._clearLine()

        
    def _clearLine(self):
        """
        Clear input line buffer
        """
        self.line.truncate(0)
        self.point = 0

        
    def _insertText(self, text):
        """
        Insert text at the current cursor position.
        """

        self.line.insert(self.point, text)
        self.point += text.length()

        cursor = self.textCursor()
        cursor.insertText(text)
        self.color_line()


    def keyPressEvent(self, e):
        """
        Handle user input a key at a time.
        """
        text  = e.text()
        key   = e.key()
        self.setUndoRedoEnabled(True)
        if key == Qt.Key_Backspace:
            if self.point:
                cursor = self.textCursor()
                cursor.movePosition(QTextCursor.PreviousCharacter, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                self.color_line()
            
                self.point -= 1 
                self.line.remove(self.point, 1)

        elif key == Qt.Key_Delete:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.NextCharacter, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self.color_line()
                        
            self.line.remove(self.point, 1)
            
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self.write('\n')
            if self.reading:
                self.reading = 0
            else:
                self._run()
                self.setUndoRedoEnabled(False)
                
        elif key == Qt.Key_Tab:
            self._insertText(text)
        elif key == Qt.Key_Left:
            if self.point : 
                self.moveCursor(QTextCursor.Left)
                self.point -= 1 
        elif key == Qt.Key_Right:
            if self.point < self.line.length():
                self.moveCursor(QTextCursor.Right)
                self.point += 1 

        elif key == Qt.Key_Home:
            cursor = self.textCursor ()
            cursor.setPosition(self.cursor_pos)
            self.setTextCursor (cursor)
            self.point = 0 

        elif key == Qt.Key_End:
            self.moveCursor(QTextCursor.EndOfLine)
            self.point = self.line.length() 

        elif key == Qt.Key_Up:
            if len(self.history):
                if self.pointer == 0:
                    self.pointer = len(self.history)
                self.pointer -= 1
                self._recall()
                
        elif key == Qt.Key_Down:
            if len(self.history):
                self.pointer += 1
                if self.pointer == len(self.history):
                    self.pointer = 0
                self._recall()
        elif e.matches(QtGui.QKeySequence.Paste):
            clipboard = QtGui.qApp.clipboard()
            mimedata = clipboard.mimeData()
            if mimedata and mimedata.hasText():
                self.moveCursor(QTextCursor.End)
                self._insertText(mimedata.text())
        elif e.matches(QtGui.QKeySequence.Copy):
            self.copy()
        elif e.matches(QtGui.QKeySequence.Undo):
            self.undo()
        elif e.matches(QtGui.QKeySequence.Redo):
            self.redo()
        elif text.length():
            self._insertText(text)
            return

        else:
            e.ignore()


    def _recall(self):
        """
        Display the current item from the command history.
        """
        cursor = self.textCursor ()
        cursor.select( QtGui.QTextCursor.LineUnderCursor )
        cursor.removeSelectedText()

        if self.more:
            self.write(sys.ps2)
        else:
            self.write(sys.ps1)
            

        self._clearLine()
        self._insertText(self.history[self.pointer])

        
    def contentsContextMenuEvent(self,ev):
        """
        Suppress the right button context menu.
        """
        pass

    
    def color_line(self):
        """ Color the current line """
        
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.StartOfLine)

        newpos = cursor.position()
        pos = -1
        
        while(newpos != pos):
            cursor.movePosition(QTextCursor.NextWord)

            pos = newpos
            newpos = cursor.position()

            cursor.select(QTextCursor.WordUnderCursor)
            word = str(cursor.selectedText ().toAscii())

            if(not word) : continue
            
            (R,G,B) = self.colorizer.get_color(word)
            
            format = cursor.charFormat()
            format.setForeground( QtGui.QBrush(QtGui.QColor(R,G,B)))
            cursor.setCharFormat(format)
            




class SyntaxColor:
    """ Allow to color python keywords """

    keywords = set(["and", "del", "from", "not", "while",
                "as", "elif", "global", "or", "with",
                "assert", "else", "if", "pass", "yield",
                "break", "except", "import", "print",
                "class", "exec", "in", "raise",              
                "continue", "finally", "is", "return",
                "def", "for", "lambda", "try"])

    def __init__(self):
        pass
        

    def get_color(self, word):
        """ Return a color tuple (R,G,B) depending of the string word """

        stripped = word.strip()
        
        if(stripped in self.keywords):
            return (255, 132,0) # orange
        
        elif(self.is_python_string(stripped)):
            return (61, 120, 9) # dark green
        
        else:
            return (0,0,0)

    def is_python_string(self, str):
        """ Return True if str is enclosed by a string mark """

#         return (
#             (str.startswith("'''") and str.endswith("'''")) or
#             (str.startswith('"""') and str.endswith('"""')) or
#             (str.startswith("'") and str.endswith("'")) or
#             (str.startswith('"') and str.endswith('"')) 
#             )
        return False
        
def main():        
    # Test the widget independently.
    from code import InteractiveInterpreter as Interpreter
    a = QtGui.QApplication(sys.argv)

    # Restore default signal handler for CTRL+C
    import signal; signal.signal(signal.SIGINT, signal.SIG_DFL)

    interpreter = Interpreter()
    aw = PyCute(interpreter)
    print type(aw)
    # static resize
    aw.resize(600,400)

    aw.show()
    a.exec_()


if __name__=="__main__":
    main()

    

