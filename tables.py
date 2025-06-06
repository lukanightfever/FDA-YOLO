# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 20:22:58 2023

@author: TheMoth
"""
import torch
import numpy as np
import os
import glob
import math
import random
import cv2
import matplotlib.pyplot as plt
import sys
sys.set_int_max_str_digits(0)

def nrotate(x,y,pointx,pointy,angle):
    nrx = (x-pointx)*math.cos(angle) - (y-pointy)*math.sin(angle)+pointx
    nry = (x-pointx)*math.sin(angle) + (y-pointy)*math.cos(angle)+pointy
    return nrx, nry

src_path = '/opt/data/private/yjl/yolov7/VOC/labels/train/'
load_path = '/opt/data/private/yjl/yolov7/VOC/'

def check_bio(s, t):
    if s & (1<<t) == 0:
        return False
    else:
        return True

class table(): 
    def __init__(self):
        super(table, self).__init__()  
        te = 1<<20000
        #print(int(str(te)))
        self.lx=[]
        self.ly=[]
        self.iglx=[]
        self.igly=[]
        for i in range(1024):
            self.lx.append(te)
            self.ly.append(te)
        tb = 1<<1025
        #print(bin(te))
        for i in range(20010):
            self.iglx.append(tb)
        for i in range(20010):
            self.igly.append(tb)
    
    def find_line(self, g_id):
        cx = random.randint(0, 1024)
        cnt = 0
        while check_bio(self.iglx[g_id], cx)==False:
            if cnt>30:
                cx = 1022
                break 
            cx = random.randint(0, 1024)
            cnt += 1
        cy = random.randint(0, 1024)
        cnt = 0
        while check_bio(self.igly[g_id], cy)==False:
            if cnt>30:
                cy = 1022
                break
            cy = random.randint(0, 1024)
            cnt += 1
        return cx,cy
    def find_g(self, cx, cy):
        lt = self.lx[cx] & self.ly[cy]
        
        ig1 = random.randint(0, 20000)
        while check_bio(lt, ig1)==False:
            ig1 = random.randint(0, 20000)
        
        ig2 = random.randint(0, 20000)
        while check_bio(lt, ig2)==False:
            ig2 = random.randint(0, 20000)
        
        ig3 = random.randint(0, 20000)
        while check_bio(lt, ig3)==False:
            ig3 = random.randint(0, 20000)        
        return ig1, ig2, ig3
    
    def load_table(self, path):
        p1 = path + 'mosaic_linetable.txt'
        p2 = path + 'mosaic_igtable.txt'
        
        with open(p1, encoding='utf-8') as file:
            lines = file.readlines()
        
        content = []
        for line in lines:
            line = line.strip('\n')
            if len(line)==0:
                continue
            content.append(line)

        for i in range(2048):
            if i < 1024:
                self.lx[i]=int(content[i])
            else:
                self.ly[i-1024]=int(content[i])
        
        file.close()
        
        with open(p2, encoding='utf-8') as file:
            lines = file.readlines()
        
        content=[]
        for line in lines:
            line = line.strip('\n')
            if len(line)==0:
                continue
            content.append(line)

        for i in range(40000):
            if i < 20000:
                self.iglx[i]=int(content[i])
            else:
                self.igly[i-20000]=int(content[i])        
        #print(self.igly)
        file.close()
        
        
    def build_table(self, path):
        txt_lst = glob.glob(os.path.join(path, '*.txt'))
        #c = 0
        for txt in txt_lst:
            #print(txt)
            tt = txt.split('/')
            #print(tt)
            tid = int(tt[-1][:-4])
            print(tid)
            
            with open(txt, encoding='utf-8') as file:
                lines = file.readlines()
            #print(lines)
            content = []
            for line in lines:
                line = line.strip('\n')
                if len(line)==0:
                    continue
                content.append(line)
            #c += 1
            
            ax = np.ones(1024)
            ay = np.ones(1024)
            for line in content:
            
                line = line.split(' ')
                sl = 1024
                cx = float(line[1])*sl
                cy = float(line[2])*sl
                w = float(line[3])*sl
                h = float(line[4])*sl
                ang = float(line[5])*3.14159/180
                #print(cx,cy,w,h)
                x1 = cx-w/2
                y1 = cy-h/2
                x2 = cx-w/2
                y2 = cy+h/2    
                x3 = cx+w/2
                y3 = cy+h/2    
                x4 = cx+w/2
                y4 = cy-h/2
                xx1,yy1=nrotate(x1,y1,cx,cy,ang)
                xx2,yy2=nrotate(x2,y2,cx,cy,ang)
                xx3,yy3=nrotate(x3,y3,cx,cy,ang)
                xx4,yy4=nrotate(x4,y4,cx,cy,ang)
                tx = int(min(min(xx1,xx2),min(xx3,xx4)))
                ty = int(min(min(yy1,yy2),min(yy3,yy4)))
                bx = int(max(max(xx1,xx2),max(xx3,xx4)))
                by = int(max(max(yy1,xx2),max(yy3,xx4)))
                
                for i in range(1024):
                    if i > tx and i < bx:
                        ax[i] = 0
                    if i > ty and i < by:
                        ay[i] = 0
            for i in range(1024):
                if ax[i]==1:
                    self.lx[i] |= (1<<tid)
                    self.iglx[tid] |= (1<<i)
            #if tid==217:
            #    print(bin(self.iglx[217]))
            
            for i in range(1024):
                if ay[i]==1:
                    self.ly[i] |= (1<<tid)
                    self.igly[tid] |= (1<<i)
        
        dst_file = '/opt/data/private/yjl/yolov7/VOC/mosaic_linetable.txt'
        f = open(dst_file,'w')
        for i in range(1024):
            f.write(str(self.lx[i])+'\n')
        for i in range(1024):
            f.write(str(self.ly[i])+'\n')
        f.close()
        
        dst_file = '/opt/data/private/yjl/yolov7/VOC/mosaic_igtable.txt'
        f = open(dst_file,'w')
        
        for i in range(20000):
            f.write(str(self.iglx[i])+'\n')
        for i in range(20000):
            f.write(str(self.igly[i])+'\n')
        f.close()
        

    
    def count(self):
        
        print(self.lst)
    
    def __len__(self): 
        return len(self.img_paths)
    
def load_mosaic(index, x, path): 
    p0 = path + str(index)+'.jpg'
    img0 = cv2.imread(p0)  # BGR    
    h0, w0 = img0.shape[:2]  # orig hw
    r = 1024 / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    #print(img.shape)
    
    cx, cy = x.find_line(index)
    ig1, ig2, ig3 = x.find_g(cx, cy)
    
    p1 = path + str(ig1)+'.jpg'
    p2 = path + str(ig2)+'.jpg'
    p3 = path + str(ig3)+'.jpg'
    
    img1 = cv2.imread(p1)  # BGR    
    img2 = cv2.imread(p2)  # BGR        
    img3 = cv2.imread(p3)  # BGR        
    
    h1, w1 = img1.shape[:2]  # orig hw
    h2, w2 = img2.shape[:2]  # orig hw
    h3, w3 = img3.shape[:2]  # orig hw
    
    r = 1024 / max(h1, w1)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img1 = cv2.resize(img1, (int(w1 * r), int(h1 * r)), interpolation=interp)    
    
    r = 1024 / max(h2, w2)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img2 = cv2.resize(img2, (int(w2 * r), int(h2 * r)), interpolation=interp)       
    
    r = 1024 / max(h3, w3)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img3 = cv2.resize(img3, (int(w3 * r), int(h3 * r)), interpolation=interp)       
    
    imgf = img0.copy()
    imgf -= imgf
    
    imgf[:cy,:cx]=img0[:cy,:cx]
    imgf[cy:h0,:cx]=img1[cy:h0,:cx]
    imgf[cy:h0,cx:w0]=img2[cy:h0,cx:w0]
    imgf[:cy,cx:w0]=img3[:cy,cx:w0]
    #print(imgf.shape) 
    plt.imshow(imgf/255)
    #print(ig1, ig2, ig3)
    
if __name__ == '__main__':
    x = table()
    x.build_table(path=src_path)
    
    
    #x.load_table(path=load_path)
    #img_path = 'D:/DatasetTransform/LabelmeTov5/VOC/images/train/'
    #load_mosaic(217, x, path = img_path)
    #x.count()
    
    
    