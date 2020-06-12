#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:07:58 2020

@author: kaushik
"""

import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"/home/kaushik/Pictures"
img = Image.open("Friends.jpg")

text = pytesseract.image_to_string(img)
print(text)