# Help.py --- 
# 
# Filename: Help.py
# Description: 
# Author: Gael Jalowicki
# Maintainer: 
# Created: Sat Jul 17 11:57:39 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 20 17:42:56 2010 (+0530)
#           By:
#     Update #:
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 


class Help(object):

	def __init__(self):
		self.text = """<html><body>
<b>About Michaelis Menten</b>
<p align="justify">
The following Moose demo enables the user to study more easily the behavior of this model by modifying its parameters and plotting the concentrations. The Michaelis Menten model is one of the most simple model for describing an enzyme kinetics. It rests on two assumptions:
Firstly the [ES] changes are much more slowly than the others two ones [S] and [P], it is called the <em>quasi-steady-state assumption</em>.
Secondly the total concentration of enzyme [E]<sub>0</sub> do <em>not</em> change over time.<br>
<br>
<b>The Michaelis Menten reaction and its parameters</b>
<p align="center">
<font color="#ff0000">
<pre><b>
 k<sub>1</sub>        k<sub>3</sub>
E + S &#60;&#61;&#61;&#61;&#61;&#62; ES ----&#62; P + E
k<sub>2</sub>         
</b></pre>
<font color="#000000">
<p align="justify">
Here k<sub>1</sub> is the forward reaction constant, k<sub>2</sub> the backward one, k<sub>3</sub> is the catalytic constant, more precisely k<sub>3</sub> is reflecting the turnover of the enzyme, in other terms the number of subtrate molecules converted to product at maximum efficiency.<br>
k<sub>1</sub> takes per time per concentration units (uM<sup>-1</sup>.s<sup>-1</sup>) while k<sub>2</sub> and k<sub>3</sub> take per time units (s<sup>-1</sup>).<br>
<br>
<b>The Michaelis Menten equation</b>
<p align="center">
<font color="#ff0000">
<pre><b>
  Vmax.[S]          k<sub>2</sub> + k<sub>3</sub>
v<sub>o</sub> =--------- and Km =--------  
Km + [S]             k<sub>1</sub>
</b></pre>
<p align="left">
<font color="#000000">
with Vm = [E]<sub>0</sub>.k<sub>cat</sub>; k<sub>cat</sub>=k<sub>3</sub> and ratio = k<sub>2</sub>/k<sub>3</sub><br>
<br><b>Setting the parameters</b>
<p align="justify">
This demo provides two ways to set the michaelis parameters. The default mode is the <em>Kinetics mode</em> which enables the user to set the k<sub>1</sub>, k<sub>2</sub> and k<sub>3</sub> constants. The second one <em>Michaelis mode</em> is based on the michaelis equation parameters Km, Vm and ratio.<br>
<br><em>Have you noticed?</em><br>
Clicking on the reset button calculates the only readeable parameters.<br>
<br>
<b>About the Michaelis Menten Enzyme Moose Object</b>
<p align="justify">
Notice that two objects are needed to describe an enzyme: an Molecule object linked to the concentration parameter and an enzyme object representing an enzyme site which has the enzyme parameters. This latter is associated to the Molecule object.<br>
In Moose, a Michaelis enzyme has two different possible forms. Simulating an <em>implicit form</em> of the enzyme means that the ES complex molecule is <em>not</em> created. This allows the level of free enzyme molecule to be <em>unaffected</em> even saturation. However, other reactions involving the enzyme do see the <em>entire</em> enzyme concentration.<br>
The <em>explicit form</em> of the enzyme, making the complex <em>apparent</em>, is maybe more realistic in terms of enzyme saturation. However the complex molecule is <em>unable</em> to participate to other reactions which is, actually, a poor assumption.
</body></html>
				"""


#
# Help.py ends here
