import matplotlib.pyplot as plt


X1 = [1,2,3,4,5,6,7,8,9]
Y1 = [0.475,0.396,0.443,0.462,0.409,0.426,0.464,0.462,0.423]

plt.figure()
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# X1的分布
# plt.plot(X1, Y1, color="#000000ff", marker='.', linestyle="-")

plt.plot(X1[0], Y1[0],label='p+n+i,r/all p+3*num(p)n', color="#00ff00", marker='*',markersize=9)
plt.plot(X1[1], Y1[1],label='p+n+i,c', color="#0000FF", marker='v',markersize=9)
plt.plot(X1[2], Y1[2],label='p+n+i,e', color="#FFFF00", marker='o',markersize=9)
plt.plot(X1[3], Y1[3], label='p+n,r',color="#FF3B1D", marker='D',markersize=9)
plt.plot(X1[4], Y1[4], label='p+n,c',color="#13C4A3", marker='>',markersize=9)
plt.plot(X1[5], Y1[5], label='p+n,e',color="#FF00FF", marker='h',markersize=9)
plt.plot(X1[6], Y1[6],label='24p+72n', color="#F4A460", marker='<',markersize=9)
plt.plot(X1[7], Y1[7],label='all p+all n', color="#b03060", marker='+',markersize=9)
plt.plot(X1[8], Y1[8],label='24p+ all n', color="#FF4500", marker='p',markersize=9)

for x,y in zip(X1,Y1):
    plt.text(x,y,str(y),ha='center',va='bottom',fontsize=10)

# X2的分布
#plt.plot(X2, Y2,label="X2",  color="#3399FF", marker='o', linestyle="-")

# X3的分布
#plt.plot(X3, Y3, label="X3",  color="#F9A602", marker='s', linestyle="-")

# X4的分布
#plt.plot(X4, Y4, label="X4",  color="#13C4A3", marker='d', linestyle="-")
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))   # 将横坐标的值全部显示出来

X_labels = ['(a)/③','(b)','(c)','(d)','(e)','(f)','①','②','④']

plt.xticks(X1,X_labels,rotation=0)
# plt.subplot(figsize=(10,5))
plt.legend(bbox_to_anchor=(0.08,0.117,0.4,0.43))

#plt.title("不同loss的mAP值",fontsize = 15)
# plt.xlabel("Type of loss function",fontsize = 15)
plt.axis([0,9.5,0.2,0.5])
plt.xlabel("Type of sample label assignment method and Sample Numbers",fontsize = 13,)
# plt.xlabel("Type of sample label assignment method",fontsize = 15)
plt.ylabel("Expected Average Overlap",fontsize = 15)
plt.yticks((0.2,0.5))

plt.vlines(6.5,ymin=0,ymax=8,color='#000000',linestyles='dashed')

# plt.xticks((0,50))
plt.savefig("折线图.jpg")
plt.show()


