import glob
import os
import sys
from pathlib import Path
from wand.image import Image 


if __name__ == "__main__":
    
    folder = str(sys.argv[1])
    destination = str(sys.argv[2])
    #print(folder)
    # Check whether the specified path exists or not
    isExist = os.path.exists(destination)

    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(destination)
      print("The new directory is created!")

    images = [file for file in glob.glob(folder+'*.jpg')]
    y=0
    x=0
    h=600
    w=1024
    width = 1024
    height = 768
    dim = (width, height)
    for i in images:
         with Image(filename = str(Path(i))) as Sampleimg:
            Sampleimg.crop(x,y,w,h)
            Sampleimg.resize(width,height)
            Sampleimg.format = 'tiff' 
            Sampleimg.save(filename =destination+str(Path(i).stem)+".tiff")


