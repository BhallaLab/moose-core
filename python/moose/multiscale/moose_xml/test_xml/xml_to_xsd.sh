#!/bin/bash
if [ $# -lt 1 ]; then 
  echo "Usage : ./xml_to_xsd.sh file.xml"
  exit 
fi 
xmlName=$1
xsdName=${xmlName/".xml"/".xsd"}
echo "Converting $xmlName to $xsdName ..."
trang $xmlName $xsdName 
