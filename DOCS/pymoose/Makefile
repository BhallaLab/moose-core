##
## Makefile
##  
## Made by Subhasis Ray
## Login   <subhasis at ncbs dot res dot in>
##
## Started on  Thu Aug 19 08:27:18 2010 Subhasis Ray
## Last update Thu Aug 19 08:27:18 2010 Subhasis Ray
## 
TARGET=pymoose.info
SOURCE=pymoose.texi
TARGET: $(SOURCE)
	makeinfo --fill-column=70 pymoose.texi

default: TARGET

clean:
	rm *.info *.tp *.vr *.aux *.fn *.fns *.pg *.log *.toc
