import biogeme as bio
from unidecode import unidecode


versionDate = "Today, 2019"
versionMajor = 3
versionMinor = 1
versionRelease = '2'
author = "Michel Bierlaire"
urlAuthor = "http://people.epfl.ch/michel.bierlaire"
department = "Transport and Mobility Laboratory"
urlDepartment = "http://transp-or.epfl.ch"
university = "Ecole Polytechnique Fédérale de Lausanne (EPFL)"
urlUniversity = "http://www.epfl.ch"
urlBiogeme = "http://biogeme.epfl.ch"
urlForum = "https://groups.google.com/d/forum/biogeme"

def getVersion():
    return "{}.{}.{}".format(versionMajor,versionMinor,versionRelease)

__version__ = getVersion()

def getHtml():
    h = "<p>biogeme "+getVersion()+" ["+versionDate+"]</p>\n"
    h += "<p><a href='https://www.python.org/' target='_blank'>Python</a> package</p>\n"
    h += "<p>Home page: <a href='"+urlBiogeme+"' target='_blank'>"+urlBiogeme+"</a></p>\n"
    h += "<p>Submit questions to <a href='"+urlForum+"' target='_blank'>"+urlForum+"</a></p>\n"
    h += "<p><a href='"+urlAuthor+"'>"
    h += author
    h += "</a>, <a href='"
    h += urlDepartment
    h += "'>"
    h += department
    h += "</a>, <a href='"
    h += urlUniversity
    h += "'>"
    h += university.encode('ascii', 'xmlcharrefreplace').decode()
    h += "</a></p>\n"
    return h

def getText():
    h = "biogeme "+getVersion()+" ["+versionDate+"]\n"
    h += "Version entirely written in Python\n"
    h += "Home page: "+urlBiogeme+"\n"
    h += "Submit questions to "+urlForum+"\n"
    h += author+", "+department+", "+university+"\n"
    return h


def getLaTeX():
    h = "%% biogeme "+getVersion()+" ["+versionDate+"]\n"
    h += "%% Version entirely written in Python\n"
    h += "%% Home page: "+urlBiogeme+"\n"
    h += "%% Submit questions to "+urlForum+"\n"
    h += "%% "+author+", "+department+", "+university+"\n"
    return h

