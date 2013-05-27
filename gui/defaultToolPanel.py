# defaultToolPanel.py --
# Gives all the default tool panel list to be displayed at mooseGui for default plugin

import moose

class DefaultToolPanel():
     ignorebaseClass =['Neutral','Panel','Msg','ZPool','none'] #ZPool b'cos ZBufPool,ZFuncPool are derived from ZPoolBaseclass
     ignored =['ZPool','ZReac','ZMMenz','ZEnz','CplxEnzBase','MarkovSolver','GslStoich','GssaStoich']
     defaultToolPanellist = []
     for ch in moose.element('/classes').children:
          #Ignore all the items which has present in 'ignorebaseClass' list and 
          #  individual items that are present in 'ignored' list and all items start with 'Zombie'
          if (ch[0].baseClass not in ignorebaseClass and 
              ch[0].name not in ignored              and 
              'Zombie' not in ch[0].name               ):

               defaultToolPanellist.append(ch)



if __name__ == '__main__':
    widget = DefaultToolPanel()
