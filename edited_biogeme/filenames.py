from pathlib import Path

def getNewFileName(name,ext):
    """
    Returns name.ext is the file does not exists.  If it does, returns
    name~xx.ext, where xx is the smallest integer such that the
    corresponding file does not exist. It is designed to avoid erasing
    output files inadvertently.
    """
    fileName = (name+"."+ext)
    theFile = Path(fileName)
    number = int(0)
    while theFile.is_file():
        fileName = "{}~{:02d}.{}".format(name,number,ext)
        theFile = Path(fileName)
        number += 1
    return fileName
