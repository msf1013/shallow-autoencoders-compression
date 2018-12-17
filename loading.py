from PIL import Image
import numpy as np
import math

def get_image_blocks_lstm(file_name, h, w):
        im = Image.open(file_name)
        height, width = im.size
        pixels = im.load()
        no_of_blocks = math.ceil((height / h) * (width / w))
        
        blocks = np.empty([no_of_blocks, 1, h, w, 3])

        c = 0
        
        for j in range(0, height, h):
                for i in range(0, width, w):
                        block = get_block(j, i, pixels, h, w)
                        blocks[c][0] = block
                        c = c + 1
        
        return blocks, im.size

def get_image_blocks(file_name, h, w):
        im = Image.open(file_name)
        height, width = im.size
        pixels = im.load()
        no_of_blocks = math.ceil((height / h) * (width / w))
        
        blocks = np.empty([no_of_blocks, h, w, 3])

        c = 0
        
        for j in range(0, height, h):
                for i in range(0, width, w):
                        block = get_block(j, i, pixels, h, w)
                        blocks[c] = block
                        c = c + 1

        return blocks, im.size
        
def get_block(y, x, pixels, h, w):
        block = np.empty([h, w, 3])
        
        for j in range(y, y + h):
                for i in range(x, x + w):
                        block[j - y][i - x][0] = (pixels[j, i][0])
                        block[j - y][i - x][1] = (pixels[j, i][1])
                        block[j - y][i - x][2] = (pixels[j, i][2])
        
        return block
        
