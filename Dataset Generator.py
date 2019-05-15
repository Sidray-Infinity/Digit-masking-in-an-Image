


from PIL import Image, ImageDraw, ImageFont
import ttfquery.findsystem 
import string
import ntpath
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

fontSize = 21
imgSize = (28,28)
position = (0,0)
 

dataset_path = os.path.join (os.getcwd(), 'Synthetic_dataset')
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
 
valid_path = os.path.join (dataset_path, 'Validation')
if not os.path.exists(valid_path):
    os.makedirs(valid_path)

test_path = os.path.join (dataset_path, 'Test')
if not os.path.exists(test_path):
    os.makedirs(test_path)
    
test_img = os.path.join (test_path, 'Test_img')
if not os.path.exists(test_img):
    os.makedirs(test_img)
    
train_path = os.path.join (dataset_path, 'Train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
    
fhandle = open('Font_list_test.txt', 'r')
upper_case_list = list(string.ascii_uppercase)
digits = range(0,10)
 
digits_list=[]
for d in digits:
    digits_list.append(str(d))
 
all_char_list = upper_case_list + digits_list
 
fonts_list = []
for line in fhandle:
    fonts_list.append(line.rstrip('\n'))
 
total_fonts = len(fonts_list)
all_fonts = glob.glob("C:\\Windows\\Fonts\\*.ttf")

f_flag = np.zeros(total_fonts)

for sys_font in all_fonts:

    font_file = ntpath.basename(sys_font)
    font_file = font_file.rsplit('.')
    font_file = font_file[0]
    f_idx = 0
    for font in fonts_list:
        f_lower = font.lower()
        s_lower = sys_font.lower()
        

        if f_lower in s_lower:
            path = sys_font
            font = ImageFont.truetype(path, fontSize)
            f_flag[f_idx] = 1
            for ch in all_char_list:
                
#                 ch_path = os.path.join (valid_path, ch)
#                 if not os.path.exists(ch_path):
#                     os.makedirs(ch_path)
                
                image = Image.new("RGB", imgSize, (255,255,255))
                draw = ImageDraw.Draw(image)
                pos_x = 7
                pos_y = 4
                pos_idx=0
                
                for y in [pos_y-1, pos_y, pos_y+1]:
                    for x in [pos_x-1, pos_x, pos_x+1]:
                        position = (x,y)
                        draw.text(position, ch, (0,0,0), font=font)
                        l_u_d_flag = "u"
                        if ch.islower():
                            l_u_d_flag = "l"
                        elif ch.isdigit():
                            l_u_d_flag = "d"
                            
                        file_name = font_file + '_' + l_u_d_flag + '_' + str(pos_idx) + '_' + ch + '.jpg'
                        file_name = os.path.join(test_img,file_name)
                        image.save(file_name)
                        pos_idx = pos_idx + 1
                        
            f_idx = f_idx + 1




