from PIL import Image
import os, sys


image = Image.open(sys.argv[1])
filename = sys.argv[2]
image = image.resize((256,256),Image.ANTIALIAS)
image.save(fp=filename)
