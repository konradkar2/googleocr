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
THRESHOLD = 0.7
DEBUG = 0
SKIP = 1 #set this to true to ommit getting data for exisitng label
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

def convert_box_to_darknet_format(imw,imh,xmin,ymin,xmax,ymax):
    x = (xmin + xmax)/2
    x = x/imw
    y = (ymin + ymax)/2
    y = y/imh

    h = (ymax-ymin)/imh
    w = (xmax-xmin)/imw
    
    return x, y, w, h

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
                        
                        ymin = 10000
                        ymax = 0
                        xmin = 10000
                        xmax = 0
                        for v in symbol.bounding_box.vertices:
                            if v.x < xmin:
                                xmin = v.x
                            if v.x > xmax:
                                xmax = v.x
                            if v.y < ymin:
                                ymin = v.y
                            if v.y > ymax:
                                ymax = v.y                        
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        #print(xmax,xmin,ymin,ymax)
                        drkn = convert_box_to_darknet_format(width,height,xmin,ymin,xmax,ymax)
                        #print(drkn)
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

        if SKIP:
            if os.path.isfile(outputpath): 
                i = i +1
                continue
       
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