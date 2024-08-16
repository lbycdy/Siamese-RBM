import matplotlib.pyplot as plt

X1 = [1,2,3,4,5,6]
Y1 = [0.488,0.493,0.452,0.480,0.461,0.513]


plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
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
plt.plot(X1[0], Y1[0],label='①', color="#00ff00", marker='*',markersize=9)
plt.plot(X1[1], Y1[1],label='②', color="#0000FF", marker='v',markersize=9)
plt.plot(X1[2], Y1[2],label='③', color="#FFFF00", marker='o',markersize=9)
plt.plot(X1[3], Y1[3], label='④',color="#13C4A3", marker='+',markersize=9)
plt.plot(X1[4], Y1[4], label='⑤',color="#FF00FF", marker='D',markersize=9)
plt.plot(X1[5], Y1[5], label='ours',color="#FF0000", marker='h',markersize=9)
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
X_labels = ['method-1','method-2','method-3','method-4','method-5','method-6']
plt.xticks(X1,X_labels,rotation=0)
# plt.subplot(figsize=(10,5))
plt.legend(bbox_to_anchor=(1.0,0.38))
# set(gca,'FontName','Times New Roman','FontSize',7,'LineWidth',1.5)
#plt.title("不同loss的mAP值",fontsize = 15)
plt.xlabel("Feature fusion and freezing parameter methods",fontsize = 15)
plt.axis([0.5,6.5,0.45,0.5])
# plt.xlabel("Sample Numbers",fontsize = 15,)
# plt.xlabel("Type of sample label assignment method",fontsize = 15)
plt.ylabel("Expected Average Overlap",fontsize = 15)
plt.yticks((0.4,0.55))
# plt.xticks((0,50))
plt.savefig("折线图.jpg")
plt.show()


