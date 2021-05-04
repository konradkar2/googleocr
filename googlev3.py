import argparse
from enum import Enum
import io
import os
from google.cloud import vision
from PIL import Image, ImageDraw
import os

credential_path = "C:/Users/salon/Documents/projects/googlekey.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

IMAGES_PATH = "ocr/images"
ANNOTATIONS_PATH = "ocr/annotations"
THRESHOLD = 0.9
DEBUG = 0

def getDict(path,debug=0):
    f = open(path, "r")
    Dict = {}
    lines = f.readlines()
    i = 0
    for line in lines:
        Dict[line[0]] = i
        i = i + 1
    if debug:
        print(Dict)
    return Dict

def convert_box_to_darknet_format(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h

def detect_document(imagepath,outputpath,Dict,threshold, debug =0):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(imagepath, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    imgtemp = Image.open(imagepath)
    width, height = imgtemp.size
    #print(width,height)
    annotList = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            #print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:              
                for word in paragraph.words:
                    #word_text = ''.join([
                    #    symbol.text for symbol in word.symbols
                    #])
                    #print('Word text: {} (confidence: {})'.format(
                       # word_text, word.confidence))

                    for symbol in word.symbols:
                        if debug:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))
                        xmin = symbol.bounding_box.vertices[0].x
                        xmax = symbol.bounding_box.vertices[1].x
                        ymin = symbol.bounding_box.vertices[0].y
                        ymax = symbol.bounding_box.vertices[2].y
                        b = (xmin,xmax,ymin,ymax)
                        drkn = convert_box_to_darknet_format((width,height),b)
                        sym = symbol.text
                        if sym in Dict.keys() and symbol.confidence > threshold:
                            result = "{} {} {} {} {} ".format(Dict[sym],*drkn)
                            annotList.append(result)
    return annotList

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


if __name__ == '__main__':
    DICT = getDict("charlabels.txt", DEBUG)
    arr = os.listdir(IMAGES_PATH)
    total = len(arr)
    i = 0
    for imgname in arr:        
        imgpath = os.path.join(IMAGES_PATH,imgname)
        pre, ext = os.path.splitext(imgname)
        outputname = pre + ".txt"
        outputpath = os.path.join(ANNOTATIONS_PATH,outputname)            
       
        annotations = detect_document(imgpath,"text", DICT,THRESHOLD, DEBUG)
        if len(annotations) > 0:
            f = open(outputpath, "w")
            f.write("\n".join(annotations))
            f.close()
        else:
            os.remove(imgpath)
        progress = str(i) + "/" + str(total)
        print ("\r Progress {}".format(progress), end="")
        
        i = i + 1