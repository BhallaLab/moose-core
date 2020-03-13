<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="xml" indent="yes" encoding="UTF-8" />
    <xsl:strip-space elements="*"/>

    <xsl:template match="/">
        <mooseml><xsl:apply-templates select="*" /></mooseml>
    </xsl:template>

    <!-- Object of Element property -->
    <xsl:template match="*">
        <xsl:value-of select="name()"/>
        <properties>
        <xsl:call-template name="Properties" />
        </properties>
    </xsl:template>

    <!-- Object Properties -->
    <xsl:template name="Properties">
        <xsl:variable name="childName" select="name(*[1])"/>
        <xsl:choose>
            <xsl:when test="not(*|@*)">"<xsl:value-of select="."/>
            </xsl:when>
            <xsl:when test="count(*[name()=$childName]) > 1">
                <xsl:value-of select="$childName"/> :[<xsl:apply-templates select="*" mode="ArrayElement"/>] 
            </xsl:when>
            <xsl:otherwise>
                <xsl:apply-templates select="@*"/>
                <xsl:apply-templates select="*"/>
            </xsl:otherwise>
        </xsl:choose>

        <xsl:if test="following-sibling::*">
        
        </xsl:if>
    </xsl:template>


</xsl:stylesheet>
