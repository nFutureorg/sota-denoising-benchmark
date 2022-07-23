import gdown

url = 'https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing'
output = 'model_zoo.zip'
gdown.download(url, output, quiet=False)
