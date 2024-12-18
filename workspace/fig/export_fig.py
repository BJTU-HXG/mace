
import numpy as np
import matplotlib.pyplot as plt  
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']

def export_fig1():
    x=[10,20,30,50,80,100,150,200,300,500,800,1200]
    x_uniform = range(len(x))

    y1=[1.692429022, 2.113207547, 2.245231608, 2.022408964, 2.188287922, 2.673643136, 7.258159509, 8.616188452, 8.732273986, 10.44455262, 7.410469629, 9.650527987]

    y2=[0.690032154, 1.174140944, 1.305174234, 1.874350987, 3.386406545, 3.286960314, 10.25554785, 15.33638083, 17.95964348, 22.69917292, 22.59647428, 27.34190458]

    y3=[1.380952381, 2.561626429, 3.117276166, 4.58703939, 6.67617866, 9.091676719, 24.85462185, 34.42816702, 37.95866039, 42.3440976, 44.54748869, 48.45832611]

    plt.plot(x_uniform,y1,'ro-',label='Scalar-Threads')
    plt.plot(x_uniform,y2,'g*-',label='HVX-MEM-Threads')
    plt.plot(x_uniform,y3,'bx-',label='HVX-REG-Threads')
    
    # 标注峰值数据
    max_y1 = max(y1)
    max_x1 =x[y1.index(max_y1)]
    max_y2 = max(y2)
    max_x2 = x[y2.index(max_y2)]
    max_y3 = max(y3)
    max_x3 = x[y3.index(max_y3)]
    # 添加峰值坐标
    plt.annotate(f'({max_x1}, {max_y1:.2f})', 
                xy=(x_uniform[y1.index(max_y1)], max_y1), 
                xytext=(x_uniform[y1.index(max_y1)]-1, max_y1+1),
                fontsize=8, color='red')
    plt.annotate(f'({max_x2}, {max_y2:.2f})', 
                xy=(x_uniform[y2.index(max_y2)], max_y2), 
                xytext=(x_uniform[y2.index(max_y2)]-1.5, max_y2+1),
                fontsize=8, color='green')
    plt.annotate(f'({max_x3}, {max_y3:.2f})', 
                 xy=(x_uniform[y3.index(max_y3)], max_y3), 
                 xytext=(x_uniform[y3.index(max_y3)]-2, max_y3+1),
                 fontsize=8, color='blue')
    # 添加峰值标线
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    # 通过设置 x_min, y_min 来保证标线从坐标轴原点（x_min, y_min）开始
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    
    plt.plot([x_uniform[y1.index(max_y1)],x_uniform[y1.index(max_y1)]], [y_min,max_y1], color='red',linestyle='--',linewidth=0.8)
    plt.plot([x_min,x_uniform[y1.index(max_y1)]],[max_y1,max_y1],color='red',linestyle='--',linewidth=1.0)
    plt.plot([x_uniform[y2.index(max_y2)],x_uniform[y2.index(max_y2)]],[y_min,max_y2],color='green',linestyle='--',linewidth=0.8)
    plt.plot([x_min,x_uniform[y2.index(max_y2)]],[max_y2,max_y2],color='green',linestyle='--',linewidth=1.0)
    plt.plot([x_uniform[y3.index(max_y3)],x_uniform[y3.index(max_y3)]],[y_min,max_y3],color='blue',linestyle='--',linewidth=1.0)
    plt.plot([x_min,x_uniform[y3.index(max_y3)]],[max_y3,max_y3],color='blue',linestyle='--',linewidth=1.0)

    
    plt.xlabel('N')
    plt.ylabel('Speed Up(x)')

    plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')  # 主网格线
    plt.grid(which='minor', linestyle=':', linewidth=0.3, color='lightgray')  # 次网格线
    plt.tick_params(axis='both', direction='in') 
    plt.xticks(ticks=x_uniform, labels=x)
    plt.legend(loc='best')

    plt.savefig('/home/ana/hhj/mace/workspace/fig/fig1.png',dpi=500)
    
def export_fig2():
    categories = ['Matmul', 'Requantize', 'Div', 'Transpose', 'Add', 'Softmax', 'Reshape']
    values = [54, 33, 4, 3, 3, 3, 1]
    colors = ['#FF9911', '#88AAFF', '#66CC99', '#FFCC99', '#CCCCFF', 'grey', '#998811']
    # 创建柱状图
    bars = plt.bar(categories, values, color=colors)
    for i, bar in enumerate(bars):
        bar.set_label(f"{categories[i]}:{values[i]}%")  # 为每个柱子设置标签
    
    # 显示图例
    plt.legend(bbox_to_anchor=(1, 1))
    
    plt.xticks(rotation=-8)  # 将横坐标标签旋转45度
    plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')  # 主网格线
    plt.grid(which='minor', linestyle=':', linewidth=0.3, color='lightgray')  # 次网格线
    
    plt.tick_params(axis='both', direction='in') 
    plt.xlabel('Op Names')
    plt.ylabel('Latency Percentage(%)')
    plt.savefig('/home/ana/hhj/mace/workspace/fig/fig2.png', dpi=500)
    
def export_fig3():
    x=[10,30,80,150,300,500]
    x_uniform = range(len(x))

    y1=[1.383627608, 2.027118644, 2.212754261, 2.333633904, 2.443178118, 2.58891225]
    y2=[1.427152318, 2.195838433, 2.751196172, 3.201649485, 3.139987445, 3.347067239]
    y3=[1.837953092, 3.38490566, 6.173312883, 9.112676056, 11.74178404, 11.81616162]

    y4=[5.10782241, 8.479423868, 6.948958089, 7.353176366, 9.397186044, 11.28077648]
    y5=[5.786826347, 10.65775862, 10.26955017, 13.92484577, 16.68619176, 14.84443628]
    y6=[6.51212938, 17.05241379, 26.31117021, 33.87175989, 42.15309168, 45.97381901]

    plt.plot(x_uniform,y1,'g*-',label='Scalar(N=100)')
    plt.plot(x_uniform,y2,'g^-',label='HVX-MEM(N=100)')
    plt.plot(x_uniform,y3,'gd-',label='HVX-REG(N=100)')

    plt.plot(x_uniform,y4,'r*-',label='Scalar(N=200)')
    plt.plot(x_uniform,y5,'r^-',label='HVX-MEM(N=200)')
    plt.plot(x_uniform,y6,'rd-',label='HVX-REG(N=200)')
    
    plt.xlabel('M')
    plt.ylabel('Speed Up(x)')

    plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')  # 主网格线
    plt.grid(which='minor', linestyle=':', linewidth=0.3, color='lightgray')  # 次网格线
    plt.tick_params(axis='both', direction='in') 
    plt.xticks(ticks=x_uniform, labels=x)
    plt.legend()

    plt.savefig('/home/ana/hhj/mace/workspace/fig/fig3.png',dpi=500)


def export_fig4():
    x=[10,30,80,150,300,500]
    x_uniform = range(len(x))

    y1=[1.165809769, 1.154602952, 2.022739291, 2.038658329, 2.980446927, 2.068065225]
    y2=[1.112883436, 0.96875, 1.875919568, 2.170702179, 2.861137713, 3.168421053]
    y3=[2.233990148, 2.34045584, 5.046174142, 6.823977165, 10.59432624, 11.34422111]

    y4=[1.091293322, 1.556607495, 2.163656885, 2.661003104, 4.423599321, 4.05213589]
    y5=[1.424944812, 2.243320068, 2.879459256, 4.737638162, 7.899444164, 6.988629771]
    y6=[2.033070866, 3.989888777, 7.227144204, 9.160854893, 18.79719439, 17.06857467]

    plt.plot(x_uniform,y1,'g*-',label='Scalar(N=100)')
    plt.plot(x_uniform,y2,'g^-',label='HVX-MEM(N=100)')
    plt.plot(x_uniform,y3,'gd-',label='HVX-REG(N=100)')

    plt.plot(x_uniform,y4,'r*-',label='Scalar(N=200)')
    plt.plot(x_uniform,y5,'r^-',label='HVX-MEM(N=200)')
    plt.plot(x_uniform,y6,'rd-',label='HVX-REG(N=200)')
    
    plt.xlabel('K')
    plt.ylabel('Speed Up(x)')

    plt.grid(which='major', linestyle='--', linewidth=0.5, color='gray')  # 主网格线
    plt.grid(which='minor', linestyle=':', linewidth=0.3, color='lightgray')  # 次网格线
    plt.tick_params(axis='both', direction='in') 
    plt.xticks(ticks=x_uniform, labels=x)
    plt.legend()

    plt.savefig('/home/ana/hhj/mace/workspace/fig/fig4.png',dpi=500)
    
    
export_fig3()
