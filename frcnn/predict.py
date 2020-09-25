from keras.layers import Input
from FRCNN import *
from PIL import Image

frcnn =FRCNN()

image_ids = open('test.txt').read().strip().split()

#if not os.path.exists("./input"):
#    os.makedirs("./input")
#if not os.path.exists("./input/detection-results"):
#    os.makedirs("./input/detection-results")
#if not os.path.exists("./input/images-optional"):
#    os.makedirs("./input/images-optional")

while True:
    for image_id in image_ids:
        image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".png"
        image = Image.open(image_path)
        imagenew = frcnn.detect_image(image_id,image)
        #image.save("./input/images-optional/"+image_id+".png")
        #imagenew.save("./result/"+image_id+"_result.png") #带框图像存储
        print(image_id, " done!")
"""

image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

while True:

    img = input('Input image filename:')
    print(img)
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image1(image)
        r_image.show()
        r_image.save('test3.png')
    """
frcnn.close_session()
