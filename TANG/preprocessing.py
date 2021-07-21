import os
import re
import json

def merge(path):
    text = ''
    pattern = re.compile(r'poet.tang.\d\d\d\d.json')
    for file in os.scandir(path):
        if re.match(pattern, file.name):
            with open(file.path, mode = 'r') as f:
                for item in json.loads(f.read()):
                    poem = ''.join(item['paragraphs'])
                    # print(poem)
                    poem += '\n'
                    text += poem
    with open('Poem.txt', mode = 'w') as file:
        file.write(text)

merge('json')