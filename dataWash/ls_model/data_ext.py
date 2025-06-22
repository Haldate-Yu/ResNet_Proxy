from data_manip.extraction.telemac_file import TelemacFile
import pandas as pd
import sys
import os

nflow = 0
if len(sys.argv) >= 2:
    nflow = sys.argv[1]

res = TelemacFile('res_ls.slf', access="r")
for i in range(res.ntimestep):

    if (i % 4 == 0) & (i != 0):
        # 初始化数据空间
        data = {}

        # 计算小时数
        hour = i // 4

        data['WaterDepth'] = res.get_data_value('WATER DEPTH', i)

        # 结果写入至csv
        output = pd.DataFrame(data)
        output.index = output.index + 1
        path = "../../data_500s/case" + str(nflow) + "/data_" + str(hour) + "h.csv"
        output.to_csv(path, index_label='ID')
