# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 20:43:02 2021

@author: afga8750
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:48:21 2021

@author: afga8750
"""


import numpy as np
import struct
from concurrent.futures import ThreadPoolExecutor


def error_Generate(input_data,ber,protectBit):
    BER = np.full_like(np.empty(32),fill_value=ber,dtype = np.float32) #ber で初期化
    # BER = np.zero_like(np.empty(32),dtype = np.float32)
    for i in protectBit:
         BER[i] = 0

    error = np.zeros(input_data.shape).astype(np.uint16)
    for i in range(4):
        error += ((np.random.random_sample(input_data.shape) < BER[i]) * (2 ** i)).astype(np.uint16)


    # print(error)
    return error
    
def turncate_Generate(input_data,turncate_num):    
    truncate = np.zeros(input_data.shape).astype(np.uint32)
    for i in range(32):
        if i >= turncate_num:
            truncate +=  (2 ** i)        
            #print(bin(truncate[0][0]))

    return truncate


def error_Apply(input_data,num,ber,protectBit,truncate_num):
    error = error_Generate(input_data,ber,protectBit)
    # print(input_data.shape)
    # print(bin(error[0]))
    truncate = turncate_Generate(input_data,truncate_num)
    # print(truncate)
    # t = time.time()
    
    if num == 1:
        #入力データfloat32 → byte        
        x = np.zeros(input_data.shape).astype(np.int16)
        for i,dim1 in enumerate(input_data):
            x[i] = struct.unpack('>i', struct.pack('>i', dim1))[0]
        #error を加える
        x = x ^ error
        """
        print("---------------")
        print("x:" + format(x[0],'032b'))
        print("t:" + format(truncate[0],'032b'))
        """
        x = x & truncate
        """
        print("x:" + format(x[0],'032b'))
        print("t:" + format(truncate[0],'032b'))
        print("---------------")
        """
        #byte → float'32'
        output = np.zeros(x.shape).astype(np.int16)
        for i,dim1 in enumerate(x):
            output[i] = struct.unpack('>i', struct.pack('>i', dim1))[0]
        return output
    
    
    if num == 2:
        #入力データ　float32 →　byte
        x = np.zeros(input_data.shape).astype(np.uint32)
        for i,dim1 in enumerate(input_data):
            for j,dim2 in enumerate(dim1):
                x[i][j] = struct.unpack('>I', struct.pack('>i', dim2))[0]
            
        #errorを加える
        x = x ^ error
        x = x & truncate
        
        
        output = np.zeros(x.shape).astype(np.uint32)
        for i,dim1 in enumerate(x):
             for j,dim2 in enumerate(dim1):
                 output[i][j] = struct.unpack('>i', struct.pack('>I', dim2))[0]
        return output
    if num == 3:
        # 入力データ　float32 →　byte
        x = np.zeros(input_data.shape).astype(np.uint32)

        for i, dim1 in enumerate(input_data):
            for j, dim2 in enumerate(dim1):
                for k, dim3 in enumerate(dim2):
                 x[i][j][k] = struct.unpack('>I', struct.pack('>i', dim3))[0]

        # errorを加える
        x = x ^ error
        x = x & truncate

        output = np.zeros(x.shape).astype(np.uint32)
        for i, dim1 in enumerate(x):
            for j, dim2 in enumerate(dim1):
                for k, dim3 in enumerate(dim2):
                 output[i][j][k]= struct.unpack('>i', struct.pack('>I', dim3))[0]
        return output

    if num == 4:
        # 入力データ　float32 →　byte
        x = np.zeros(input_data.shape).astype(np.uint32)

        for i, dim1 in enumerate(input_data):
            for j, dim2 in enumerate(dim1):
                for k, dim3 in enumerate(dim2):
                    for k1, dim4 in enumerate(dim3):
                     x[i][j][k][k1] = struct.unpack('>I', struct.pack('>i', dim4))[0]

        # errorを加える
        x = x ^ error

        print("---------------")
        print(x[0][0][0][0])
        print("x:" + format(x[0][0][0][0],'032b'))
        print("t:" + format(truncate[0][0][0][0],'032b'))
        
        x = x & truncate
        
        print("x:" + format(x[0][0][0][0],'032b'))
        print("t:" + format(truncate[0][0][0][0],'032b'))
        print("---------------")

        output = np.zeros(x.shape).astype(np.uint32)
        for i, dim1 in enumerate(x):
            for j, dim2 in enumerate(dim1):
                for k, dim3 in enumerate(dim2):
                    for k1, dim4 in enumerate(dim3):
                     output[i][j][k][k1] = struct.unpack('>i', struct.pack('>I', dim4))[0]
        return output


    

        
        

