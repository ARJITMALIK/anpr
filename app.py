import cv2
import numpy as np 
import imutils
import urllib
import easyocr
from matplotlib import pyplot as plt
import os

def get_from_url(uri):
    url = uri
    header= {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
        'AppleWebKit/537.11 (KHTML, like Gecko) '
        'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    # getting image from url
    request = urllib.request.Request(url, headers = header)
    req = urllib.request.urlopen(request)
    arr = np.asarray(bytearray(req.read()), dtype = np.uint8)
    img = cv2.imdecode(arr, -1)
    process_img(img)

def process_img(img):
    # converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    #finding contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # drawing contours
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    cv2.imwrite('./static/onlyPlate.png', cropped_image)

    # formulating results
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    if np.any(result):
        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        cv2.imwrite('./static/savedImg.png',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    else:
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        cv2.imwrite('./static/savedImg.png',cv2.cvtColor(res, cv2.COLOR_BGR2RGB))