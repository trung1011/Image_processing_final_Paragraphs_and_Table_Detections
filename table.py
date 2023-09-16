#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math


# ## Load image, grayscale, Otsu's threshold

# In[2]:


image = cv2.imread('table7.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bin = 255-img_bin


# ## Detect horizontal lines 

# In[3]:


# Detect horizontal lines 
kernel_len = gray.shape[1]//120
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
image_horizontal = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_horizontal, hor_kernel, iterations=3)
h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 30, maxLineGap=250)


# ## Group horizontal lines if they in the same line
# 
# ### Input: list of horizontal lines, threshold of lines(how far apart of each other)
# ### Output: new horizontal lines that are grouped

# In[4]:


def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
        lines = [line for line in h_lines if thresh[1] - thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh > line[0][1] or line[0][1] > thresh[1] + thin_thresh]
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
        x_min, x_max = min(x) - int(5*thin_thresh), max(x) + int(5*thin_thresh)
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines
    
new_horizontal_lines = group_h_lines(h_lines, kernel_len)


# ## Detect vertical lines 

# In[5]:


kernel_len = gray.shape[1]//120
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
image_vertical = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_vertical, ver_kernel, iterations=3)

v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 30, maxLineGap=250)


# ## Group vertical lines if they in the same line
# 
# ### Input: list of vertical lines, threshold of lines(how far apart of each other)
# ### Output: new vertical lines that are grouped

# In[6]:


def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines
    
new_vertical_lines = group_v_lines(v_lines, kernel_len)


# ## Find intersect point of a horizontal line and a vertical line
# 
# ### Input: two lines
# ### Output: intersect point of two input lines

# In[7]:


def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1


# ## Get intersect points

# In[8]:


points = []
for hline in new_horizontal_lines:
    x1A, y1A, x2A, y2A = hline
    for vline in new_vertical_lines:
        x1B, y1B, x2B, y2B = vline
        line1 = [np.array([x1A, y1A]), np.array([x2A, y2A])]
        line2 = [np.array([x1B, y1B]), np.array([x2B, y2B])]

        x, y = seg_intersect(line1, line2)
        if x1A <= x <= x2A and y1B <= y <= y2B:
            points.append([int(x), int(y)])

            


# ## Draw lines of tables

# In[9]:


# for i in points:
#     for j in points:
#         dis=math.sqrt( ((int(i[0])-int(j[0]))**2)+((int(i[1])-int(j[1]))**2) )
#         if (dis<150 and (j[0]-3 <= i[0] <= j[0]+3 or j[1]-3 <= i[1] <= j[1]+3)):
#             cv2.line(image, (i[0], i[1]), (j[0], j[1]), (0, 0, 255), thickness=2)


# ## Gaussian blur, Otsu's threshold

# In[10]:


blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# ## Create rectangular structuring element and dilate

# In[11]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)


# ## Find contours, draw rectangle and numbering

# In[12]:


cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
n =0
para_c=0
table_c=0
for c in cnts:
    n+=1
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    isTable=False
    for p in points:
        if (x <= p[0] <= x+w and y <= p[1] <= y+h):
            isTable=True
            
    if (w>50 and h>50):
        
        if (isTable==True):
            table_c+=1
            cv2.rectangle(image, (x+5, y+5), (x + w -5, y + h -5), (0, 0, 255), 2)
            cv2.putText(image, "Table "+str(table_c), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            para_c+=1
            cv2.rectangle(image, (x+5, y+5), (x + w -5, y + h -5), (36,255,12), 2)
            cv2.putText(image, "Paragraph "+str(para_c), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (32, 158, 68), 1, cv2.LINE_AA)
            
# cv2.imshow('thresh', thresh)
# cv2.imshow('dilate', dilate)
print("Number of paragraph: ",para_c)
print("Number of table: ",table_c)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




