import numpy as np
from PIL import Image


def save(begin, end, path, pet, dst):
    r = 80
    c = 80
    ind = 0
    for i in range(begin, end):
        img = Image.open(path + str(i) + '.jpg')
        img = img.resize((r, c))
        img = img.convert('RGB')
        arr = np.frombuffer(img.tobytes(), dtype = np.uint8)
        arr = arr.reshape((r, c, 3))
        buf = ''
        for k in range(0, 3):
            for x in range(0, r):
                for y in range(0, c):
                    buf += str(arr[x, y, k])
                    buf += ' '
                buf += '\n'
            buf += '\n'
        file = open(dst + pet + '/' + str(ind) + '.txt', 'w')
        file.write(buf)
        file.close()
        ind += 1

def load(path, train_size, test_size):
    save(0, train_size, path, path.split('/')[-2], "Train/")
    save(train_size, train_size + test_size, path, path.split('/')[-2], "Test/")
    

load('PetImages/Cat/', 10000, 2500)
load('PetImages/Dog/', 10000, 2500)