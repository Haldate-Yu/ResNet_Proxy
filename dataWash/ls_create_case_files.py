import numpy as np
import os
import scipy as sp
import pandas as pd
import shutil

io = r'natural_flow_500_24h.csv'
order_dict = pd.read_csv(io,
                         header=None,  # 指定哪几行做列名
                         usecols=None,  # 指定读取的列
                         )  # 跳过指定行
print(order_dict.items())
# path = './test/' #./指的是当前路径下的test文件夹
path = './ls_cases_500s/'
srcPath = './ls_model/'
# for col in range(order_dict.shape[1]):

for counter, df in order_dict.items():
    data = df.values

    if counter == 0:
        time_series = data
        # print(time_series)
    else:

        # print(data.shape, counter)

        # ============== 创建以sheet_name为名的文件夹 ================
        # 定义一个变量判断文件是否存在,path指代路径,str(i)指代文件夹的名字*
        isExists1 = os.path.exists(path + 'case' + str(counter))
        if not isExists1:  # 判断如果文件不存在,则创建
            os.makedirs(path + 'case' + str(counter))
            print("case%s 目录创建成功" % str(counter))
        else:
            print("case%s 目录已经存在" % str(counter))
            # pass  # 如果文件不存在,则继续上述操作,直到循环结束
        #
        namePath = path + 'case' + str(counter)
        # =========== 将文件复制到指定目录中 ================
        srcDir = [srcPath + 'source_regions.qsl',
                  srcPath + 'ls.slf',
                  srcPath + 'ls.cas',
                  srcPath + 'ls.cli',
                  srcPath + 'restart.slf',
                  srcPath + 'data_ext.py']

        # 将srcDir中的文件都复制到指定的case文件夹中
        for mm in range(len(srcDir)):
            dstDir = namePath
            shutil.copy(srcDir[mm], dstDir)

        # =========== 将洪水流量过程曲线数据按照相应格式写入到相应文件夹中 =================
        ipath = namePath + '/' + 'source_discharge.qsl'
        ##这里的a表示没有该文件就新建,文件指针自动移动到文件末尾进行添加.
        ##这里的w表示写文件，若文件不存在，则创建文件；若文件存在，先清空，再打开

        with open(ipath, "w") as f:
            f.write('T \t Q(1)\n s \t m3/s \n')

            for irow in range(data.shape[0]):
                f.write(str(time_series[irow]) + '\t' + str(data[irow]) + '\n')  # 写入文件
