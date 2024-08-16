import numpy as np
from matplotlib import pyplot as plt

from matplotlib.pyplot import MultipleLocator

# 规范化矩形描述方式
# 传入四点或两点坐标返回，一点的坐标加宽高


def get_axis_aligned_bbox(region):
    region = np.asarray(region)
    nv = len(region)
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return (cx - w / 2, cy - h / 2, w, h)
    else:
        return (region[0], region[1], region[2] - region[0], region[3] - region[1])

# print(get_axis_aligned_bbox([28.788,57.572,97.714,57.116,98.27,141.12,29.344,141.58]))

# 传入两个矩形的左下角和右上角的坐标，得出相交面积，与面积


def computeArea(rect1, rect2):
    #  让rect 1 靠左
    if rect1[0] > rect2[0]:
        return computeArea(rect2, rect1)
    # 没有重叠
    if rect1[1] >= rect2[3] or rect1[3] <= rect2[1] or rect1[2] <= rect2[0]:
        return 0, rect1[2] * rect1[3] + rect2[2] * rect2[3]
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    rect1w = rect1[2] - rect1[0]
    rect1h = rect1[3] - rect1[1]
    rect2w = rect2[2] - rect2[0]
    rect2h = rect2[3] - rect2[1]
    return abs(x1 - x2) * abs(y1 - y2), rect1w * rect1h + rect2w * rect2h - abs(x1 - x2) * abs(y1 - y2)

#print(computeArea([-3,0,3,4], [0,-1,9,2]))

# 从文件读入坐标


def readData(path, separator, need):
    reader = open(path, "r", encoding='utf-8')
    ans = []
    lines = reader.readlines()
    for i in range(len(lines)):
        t = lines[i].split(separator)
        t = [float(i) for i in t]
        if need:
            ans.append(get_axis_aligned_bbox(t))
        else:
            ans.append(t)
    return ans


def getCenter(region):
    return (region[0] + region[2] / 2, region[1] + region[3] / 2)


def computePrecision(myData, trueData, x):
    # 获取中心差
    cen_gap = []
    for i in range(len(myData)):
        x1 = myData[i][0]
        y1 = myData[i][1]
        x2 = trueData[i][0]
        y2 = trueData[i][1]
        cen_gap.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
    # 计算百分比
    precision = []
    for i in range(len(x)):
        gap = x[i]
        count = 0
        for j in range(len(cen_gap)):
            if cen_gap[j] < gap:
                count += 1
        precision.append(count/len(cen_gap))

    return precision


def computeSuccess(myData, trueData, x):
    frames = len(trueData)
    # 获取重合率得分
    overlapScore = []
    for i in range(frames):
        one = [myData[i][0], myData[i][1], myData[i][0] +
               myData[i][2], myData[i][1] + myData[i][3]]
        two = [trueData[i][0], trueData[i][1], trueData[i][0] +
               trueData[i][2], trueData[i][1] + trueData[i][3]]
        a, b = computeArea(one, two)
        overlapScore.append(a / b)

    # 计算百分比
    precision = []
    for i in range(len(x)):
        gap = x[i]
        count = 0
        for j in range(frames):
            if overlapScore[j] > gap:
                count += 1
        precision.append(count/frames)

    return precision

plt.rcParams['font.family'] = 'Microsoft YaHei'


def showSuccess(girlData,girlData1,girlData2,girlData3, girlData4, girlData5,girlData6,girlData7,girlData8, trueData):
    # 生成阈值，在[start, stop]范围内计算，返回num个(默认为50)均匀间隔的样本
    xSuccess = np.linspace(0, 1, 20)
    y1Success = []
    y2Success = []
    y3Success = []
    y4Success = []
    y5Success = []
    y6Success = []
    y7Success = []
    y8Success = []
    y9Success = []

    for i in girlData:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y1Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData1:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y2Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData2:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y3Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData3:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y4Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData4:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y5Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData5:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y6Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData6:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y7Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData7:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y8Success.append(computeSuccess(i, trueData, xSuccess))
    for i in girlData8:
        # 分别存放所有点的横坐标和纵坐标，一一对应
        y9Success.append(computeSuccess(i, trueData, xSuccess))
    # for i in girlData8:
    #     # 分别存放所有点的横坐标和纵坐标，一一对应
    #     y9Success.append(computeSuccess(i, trueData, xSuccess))
    # 创建图并命名
    plt.figure('Success plot in different algorithms')
    ax = plt.gca()
    # 设置x轴、y轴名称
    plt.xlim(0,1)
    plt.ylim(0,0.82)
    ax.set_xlabel('重叠阈值', fontsize=16)
    ax.set_ylabel('成功率', fontsize=16)

    y_major_locator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    for i in range(len(girlData)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y1Success[i], color="m", linewidth=2,
                alpha=0.6, label='SiamRBO' + '[0.532]', linestyle='--')
    for i in range(len(girlData5)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y6Success[i], color="g", linewidth=2,
                alpha=0.6, label='SiamBAN' + '[0.524]',linestyle='dashed')
    for i in range(len(girlData1)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y2Success[i], color="r", linewidth=2,
                alpha=0.6, label='Siamese-remo' + "[0.521]", linestyle='-')
    for i in range(len(girlData6)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y7Success[i], color="b", linewidth=2,
                alpha=0.6, label='SiamACM' + '[0.501]', linestyle='dotted')

    for i in range(len(girlData3)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y4Success[i], color="y", linewidth=2,
                alpha=0.6, label='OCean' + '[0.483]', linestyle='dashed')
    for i in range(len(girlData8)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y9Success[i], color="#3399FF", linewidth=2,
                alpha=0.6, label='SiamMASK' + '[0.469]', linestyle='-')

    for i in range(len(girlData4)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y5Success[i], color="c", linewidth=2,
                alpha=0.6, label='SiamRPN++' + '[0.365]', linestyle='-')
    for i in range(len(girlData7)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y8Success[i], color="#F9A602", linewidth=2,
                alpha=0.6, label='SiamRPN' + '[0.317]', linestyle='dashdot')
    for i in range(len(girlData2)):
        # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
        # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
        ax.plot(xSuccess, y3Success[i], color="#b03060", linewidth=2,
                alpha=0.6, label='SiamFC' + '[0.357]',linestyle='dashdot')


    # 设置图例的最好位置
    plt.legend(loc="best")

    plt.savefig('Success Plot.jpg')
    plt.show()
# girlData = readData(r"/home/lbycdy/precision plot/1.txt", ",", False)
# girlData1 = readData(r"/home/lbycdy/precision plot/2.txt", ",", False)
# girlData2 = readData(r"/home/lbycdy/precision plot/3.txt", ",", False)
# girlData3 = readData(r"/home/lbycdy/precision plot/4.txt", ",", False)
# girlData4 = readData(r"/home/lbycdy/precision plot/5.txt", ",", False)
# girlData5 = readData(r"/home/lbycdy/precision plot/6.txt", ",", False)
# girlDataTrue = readData(
#     r"/home/lbycdy/precision plot/groundtruth.txt", ",", False)


girlData = readData(r"/home/lbycdy/success plot1/1.txt", ",", False)
girlData1 = readData(r"/home/lbycdy/success plot1/6.txt", ",", False)
girlData2 = readData(r"/home/lbycdy/success plot1/3.txt", ",", False)
girlData3 = readData(r"/home/lbycdy/success plot1/4.txt", ",", False)
girlData4 = readData(r"/home/lbycdy/success plot1/5.txt", ",", False)
girlData5 = readData(r"/home/lbycdy/success plot1/2.txt", ",", False)
girlData6 = readData(r"/home/lbycdy/success plot1/7.txt", ",", False)
girlData7 = readData(r"/home/lbycdy/success plot1/8.txt", ",", False)
girlData8 = readData(r"/home/lbycdy/success plot1/9.txt", ",", False)
girlDataTrue = readData(
    r"/home/lbycdy/success plot1/groundtruth.txt", ",", False)
showSuccess([girlData], [girlData1], [girlData2], [girlData3], [girlData4], [girlData5],[girlData6],[girlData7],[girlData8],girlDataTrue)
# showPrecision([girlData],[girlData1],[girlData2],[girlData3],[girlData4],[girlData5], girlDataTrue)

