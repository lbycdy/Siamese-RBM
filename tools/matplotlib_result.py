import matplotlib.pyplot as plt

# X1 = [1,2,3,]
# Y1 = [0.464,0.462,0.423]
X1 = [1,2,3,4,5,6]
Y1 = [0.441,0.452,0.395,0.364,0.446,0.475]
# X1 = [1,2,3,4,5,6]
# Y1 = [0.475,0.396,0.443,0.462,0.409,0.426]
# X2 = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# Y2 = [554,861,1238,1979,3206,10663,2916,1639,1047,704,489]
# X3 = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# Y3 = [363,547,801,1254,2205,7535,1984,1115,677,464,340]
# X4 = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
# Y4 = [293,478,681,1070,1819,6563,1750,953,583,410,287]

plt.figure()
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.family'] = 'Times New Roman'
# X1的分布
# plt.plot(X1, Y1, color="#000000ff", marker='.', linestyle="-")
# plt.plot(X1[0], Y1[0],label='(a)', color="#F4A460", marker='<',markersize=9)
# plt.plot(X1[1], Y1[1],label='(b)', color="#b03060", marker='+',markersize=9)
# plt.plot(X1[2], Y1[2],label='(c)', color="#FF4500", marker='p',markersize=9)
# plt.plot(X1[0], Y1[0],label='(a)', color="#00ff00", marker='*',markersize=9)
# plt.plot(X1[1], Y1[1],label='(b)', color="#0000FF", marker='v',markersize=9)
# plt.plot(X1[2], Y1[2],label='(c)', color="#FFFF00", marker='o',markersize=9)
# plt.plot(X1[3], Y1[3], label='(d)',color="#FF3B1D", marker='D',markersize=9)
# plt.plot(X1[4], Y1[4], label='(e)',color="#13C4A3", marker='>',markersize=9)
# plt.plot(X1[5], Y1[5], label='(f)',color="#FF00FF", marker='h',markersize=9)
plt.plot(X1[0], Y1[0],label='Siamban', color="#00ff00", marker='*',markersize=9)
plt.plot(X1[1], Y1[1],label='ours', color="#0000FF", marker='v',markersize=9)
plt.plot(X1[2], Y1[2],label='Siamese_CAR', color="#FFFF00", marker='o',markersize=9)
plt.plot(X1[3], Y1[3], label='Siamese_Concat',color="#FF3B1D", marker='h',markersize=9)
plt.plot(X1[4], Y1[4], label='Siamese_Mask',color="#FF00FF", marker='D',markersize=9)
plt.plot(X1[5], Y1[5], label='Siamese_RBO',color="#13C4A3", marker='+',markersize=9)
for x,y in zip(X1,Y1):
    plt.text(x,y,str(y),ha='center',va='bottom',fontsize=10)

# X2的分布
#plt.plot(X2, Y2,label="X2",  color="#3399FF", marker='o', linestyle="-")

# X3的分布
#plt.plot(X3, Y3, label="X3",  color="#F9A602", marker='s', linestyle="-")

# X4的分布
#plt.plot(X4, Y4, label="X4",  color="#13C4A3", marker='d', linestyle="-")
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))   # 将横坐标的值全部显示出来
# X_labels = ['①','②','④']
# X_labels = ['(a)/③','(b)','(c)','(d)','(e)','(f)']
# X_labels = ['(1)','(2)','(3)','(4)','(5)','(6)']
X_labels = ['type1','type2','type3','type4','type5','type6']
plt.xticks(X1,X_labels,rotation=0)
# plt.subplot(figsize=(10,5))
plt.legend(bbox_to_anchor=(1.0,0.38))
# set(gca,'FontName','Times New Roman','FontSize',7,'LineWidth',1.5)
#plt.title("不同loss的mAP值",fontsize = 15)
plt.xlabel("不同loss方案",fontsize = 15)
plt.axis([0.5,6.5,0.35,0.5])
# plt.xlabel("Sample Numbers",fontsize = 15,)
# plt.xlabel("Type of sample label assignment method",fontsize = 15)
plt.ylabel("平均期望重叠EAO",fontsize = 15)
plt.yticks((0.35,0.5))
# plt.xticks((0,50))
plt.savefig("折线图.jpg")
plt.show()


