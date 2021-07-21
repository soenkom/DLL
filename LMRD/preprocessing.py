import re
import os

def merge(path):
    text = ''
    for file in os.scandir(path):
        with open(file.path, mode = 'r') as f:
            text += f.read()
            text += '\n'
    text = re.sub('<br />', ' ', text)
    text = re.sub(',', ' ', text)
    text = re.sub('\.', ' ', text)
    text = re.sub('<', ' ', text)
    text = re.sub('>', ' ', text)
    text = re.sub('/', ' ', text)
    text = re.sub('\?', ' ', text)
    text = re.sub(';', ' ', text)
    text = re.sub(':', ' ', text)
    text = re.sub('"', ' ', text)
    text = re.sub("'", ' ', text)
    text = re.sub('\[', ' ', text)
    text = re.sub('\]', ' ', text)
    text = re.sub('\|', ' ', text)
    text = re.sub(r'\\', ' ', text)
    text = re.sub('`', ' ', text)
    text = re.sub('~', ' ', text)
    text = re.sub('!', ' ', text)
    text = re.sub('@', ' ', text)
    text = re.sub('#', ' ', text)
    text = re.sub('\$', ' ', text)
    text = re.sub('%', ' ', text)
    text = re.sub('\^', ' ', text)
    text = re.sub('&', ' ', text)
    text = re.sub('\*', ' ', text)
    text = re.sub('\(', ' ', text)
    text = re.sub('\)', ' ', text)
    text = re.sub('-', ' ', text)
    text = re.sub('_', ' ', text)
    text = re.sub('\+', ' ', text)
    text = re.sub('=', ' ', text)
    text = re.sub(' {2,}', ' ', text)
    name = path.split('/')[-1]
    with open(name, mode = 'w') as file:
        file.write(text.lower())

merge('aclImdb/train/pos')
merge('aclImdb/train/neg')