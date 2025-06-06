# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:49:08 2023

@author: TheMoth
"""

import torch
import numpy as np
import math
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def down_(x, y, cx, cy, l):
    x = x - cx
    y = y - cy
    dis = math.sqrt(x**2 + y**2)
    cos = x / dis
    sin = y / dis
    
    dis = dis - l
    ux = round(dis * cos)
    uy = round(dis * sin)
    return ux+cx, uy+cy

def up_(x, y, cx, cy, l):
    x = x - cx
    y = y - cy
    dis = math.sqrt(x**2 + y**2)
    cos = x / dis
    sin = y / dis
    dis = dis + l
    ux = round(dis * cos)
    uy = round(dis * sin)
    return ux+cx, uy+cy 

def left_(x, y, cx, cy, r):
    x = x - cx
    y = y - cy
    dis = math.sqrt(x**2 + y**2)
    cos = x / dis
    sin = y / dis
    theta = 0
    if sin>=0 and cos>=0:
        theta = math.acos(cos)*180/math.pi
    elif sin>=0 and cos <=0:
        theta = 90+math.acos(sin)*180/math.pi
    
    elif sin<=0 and cos <=0:
        theta = 180+math.asin(-sin)*180/math.pi
    elif sin<=0 and cos >=0:
        theta = 360-math.acos(cos)*180/math.pi

    theta = theta + r
    if theta>360:
        theta -= 360
    
    theta = theta/180*math.pi
    ux = round(dis*math.cos(theta))
    uy = round(dis*math.sin(theta))
    return ux+cx,uy+cy

def right_(x, y, cx, cy, r):
    x = x - cx
    y = y - cy
    dis = math.sqrt(x**2 + y**2)
    cos = x / dis
    sin = y / dis
    theta = 0
    if sin>=0 and cos>=0:
        theta = math.acos(cos)*180/math.pi
    elif sin>=0 and cos <=0:
        theta = 90+math.acos(sin)*180/math.pi
    
    elif sin<=0 and cos <=0:
        theta = 180+math.asin(-sin)*180/math.pi
    elif sin<=0 and cos >=0:
        theta = 360-math.acos(cos)*180/math.pi

    theta = theta - r
    if theta<0:
        theta += 360
    
    theta = theta/180*math.pi
    ux = round(dis*math.cos(theta))
    uy = round(dis*math.sin(theta))
    return ux+cx,uy+cy

def show_round(le):
    x = 25
    y = 100
    b = np.zeros((128,128))
    b[x][y] = 1
    ux = le[x][y][0]
    uy = le[x][y][1]    
    for i in range(100):
        b[x][y] = 1
        ux = le[x][y][0]
        uy = le[x][y][1]
        x = round(ux)
        y = round(uy)
        print(x,y)
    plt.imshow(b)
    plt.show()
    

def go(a, le, x, y):
    
    w = a.shape[0]
    h = a.shape[1]
    cx = w // 2
    cy = h // 2
        
    scl = 360.0/(math.pi*w)*3
    for i in range(600):
        bx, by = left_(x, y, cx, cy, scl*(i))
        ux, uy = left_(x, y, cx, cy, scl*(i+1))
        if bx<0 or bx>=w or by<0 or by>=h:
            break
        a[bx][by]=1
        if ux>=0 and ux<w and uy>=0 and uy<h:
            le[bx][by][0] = round(ux)
            le[bx][by][1] = round(uy)
            if(a[ux][uy]!=0):
                break
        else:
            le[bx][by][0] = round(bx)
            le[bx][by][1] = round(by)
    return a, le

def goes(a, ri, x, y):
    w = a.shape[0]
    h = a.shape[1]
    cx = w // 2
    cy = h // 2
    scl = 360.0/(math.pi*w)*3
    for i in range(600):
        bx, by = right_(x, y, cx, cy, scl*(i))
        ux, uy = right_(x, y, cx, cy, scl*(i+1))
        if bx<0 or bx>=w or by<0 or by>=h:
            break
        a[bx][by]=1
        if ux>=0 and ux<w and uy>=0 and uy<h:
            ri[bx][by][0] = round(ux)
            ri[bx][by][1] = round(uy)
            if(a[ux][uy]!=0):
                break
        else:
            ri[bx][by][0] = round(bx)
            ri[bx][by][1] = round(by)
    return a, ri   


def ls(a, up, x, y):
    w = a.shape[1]
    h = a.shape[0]
    cx = w // 2
    cy = h // 2
    for i in range(80):
        bx, by = up_(x, y, cx, cy, i)
        ux, uy = up_(x, y, cx, cy, i+1.5)
        #print(bx, by)
        if bx<0 or bx>=w or by<0 or by>=h:
            break
        a[bx][by]=1
        if ux>=0 and ux<w and uy>=0 and uy<h:
            up[bx][by][0] = round(ux)
            up[bx][by][1] = round(uy)
            if(a[ux][uy]!=0):
                break
        else:
            up[bx][by][0] = round(bx)
            up[bx][by][1] = round(by)
    return a,up

def lss(a, down, x, y):
    w = a.shape[1]
    h = a.shape[0]
    cx = w // 2
    cy = h // 2
    for i in range(80):
        bx, by = down_(x, y, cx, cy, i)
        ux, uy = down_(x, y, cx, cy, i+1.5)
        #print(bx, by)
        if bx<0 or bx>=w or by<0 or by>=h:
            break
        down[bx][by]=1
        
        dis = math.sqrt((ux-cx)**2+(uy-cy)**2)
        if dis>30:
            down[bx][by][0] = round(ux)
            down[bx][by][1] = round(uy)
            if(a[ux][uy]!=0):
                break
        else:
            down[bx][by][0] = round(bx)
            down[bx][by][1] = round(by)
    return a, down

class round_():
    
    left = np.zeros((128,128,2))
    right = np.zeros((128,128,2))
    up = np.zeros((128,128,2))
    down = np.zeros((128,128,2))
    lw = np.zeros((128,128,2))
    ld = np.zeros((128,128,2))
    rw = np.zeros((128,128,2))
    rd = np.zeros((128,128,2))
    sp = None
    
    def __init__(self):
        a = np.zeros((128, 128))
        w = a.shape[0]
        h = a.shape[1]
        cx = w//2
        cy = h//2
        yu = 30
        for i in range(w):
            for j in range(h):
                dis = math.sqrt((i-cx)**2+(j-cy)**2)
                if dis<yu:
                    continue
                if a[i][j] == 0:
                    a, self.left = go(a, self.left, i, j)
        a = np.zeros((128, 128))
        for i in range(w):
            for j in range(h):
                dis = math.sqrt((i-cx)**2+(j-cy)**2)
                if dis<yu:
                    continue
                if a[i][j] == 0:
                    a, right = goes(a, self.right, i, j)
        a = np.zeros((128, 128))
        
        for i in range(w):
            for j in range(h):
                dis = math.sqrt((i-cx)**2+(j-cy)**2)
                if dis<yu:
                    continue
                if a[i][j] == 0:
                    a, up = ls(a, self.up, i, j)
                
        a = np.zeros((128, 128))
        for i in range(w):
            for j in range(h):
                dis = math.sqrt((i-cx)**2+(j-cy)**2)
                if dis<yu:
                    continue
                if a[i][j] == 0:
                    a, up = lss(a, self.down, i, j)
        for i in range(w):
            for j in range(h):
                dis = math.sqrt((i-cx)**2+(j-cy)**2)
                if dis<yu:
                    self.left[i][j][0] = i
                    self.left[i][j][1] = j - 1
                    self.right[i][j][0] = i
                    self.right[i][j][1] = j + 1
                    self.up[i][j][0] = i - 1
                    self.up[i][j][1] = j  
                    self.down[i][j][0] = i + 1
                    self.down[i][j][1] = j
    
        for i in range(w):
            for j in range(h):
                if(self.left[i][j][0]==0):
                    self.left[i][j][0]=i
                if(self.left[i][j][1]==0):
                    self.left[i][j][0]=j
        for i in range(w):
            for j in range(h):
                if(self.right[i][j][0]==0):
                    self.right[i][j][0]=i
                if(self.right[i][j][1]==0):
                    self.right[i][j][0]=j 
        for i in range(w):
            for j in range(h):
                if(self.up[i][j][0]==0):
                    self.up[i][j][0]=i
                if(self.up[i][j][1]==0):
                    self.up[i][j][0]=j 
        for i in range(w):
            for j in range(h):
                if(self.down[i][j][0]==0):
                    self.down[i][j][0]=i
                if(self.down[i][j][1]==0):
                    self.down[i][j][0]=j 
        
        for i in range(w):
            for j in range(h):
                l1 = int(self.left[i][j][0])
                l2 = int(self.left[i][j][1])
                w1 = int(self.up[i][j][0])
                w2 = int(self.up[i][j][1])
                    
                d1 = int(self.down[i][j][0])
                d2 = int(self.down[i][j][1])
                r1 = int(self.right[i][j][0])
                r2 = int(self.right[i][j][1])
                
                lw1 = int(self.up[l1][l2][0])
                lw2 = int(self.up[l1][l2][1])
                rw1 = int(self.up[r1][r2][0])
                rw2 = int(self.up[r1][r2][1])
                ld1 = int(self.down[l1][l2][0])
                ld2 = int(self.down[l1][l2][1])
                rd1 = int(self.down[r1][r2][0])
                rd2 = int(self.down[r1][r2][1])
                
                self.lw[i][j][0] = lw1
                self.lw[i][j][1] = lw2
                self.rw[i][j][0] = rw1
                self.rw[i][j][1] = rw2        
                self.ld[i][j][0] = ld1
                self.ld[i][j][1] = ld2
                self.rd[i][j][0] = rd1
                self.rd[i][j][1] = rd2
        
    def concatt(self):
         
        self.left = self.left[None,:,:]
        self.right = self.right[None,:,:]
        self.up = self.up[None,:,:]
        self.down = self.down[None,:,:]
        self.lw = self.lw[None,:,:]
        self.rw = self.rw[None,:,:]
        self.ld = self.ld[None,:,:]
        self.rd = self.rd[None,:,:]
        
        self.sp = np.concatenate((self.left,self.right,
                                  self.up,self.down,self.lw,self.rw,self.ld,self.rd),axis=0)
        
        print(self.sp.shape)

if __name__ == '__main__':

    round_grid = round_()
    show_round(round_grid.rw)
    
    round_grid.concatt()