import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import math
import random

### 配置文件
# 正弦图像的个数
N=100
# 幅度
A=10
# 噪声大小
Noise=20
# 正弦信号频率
fs=1/(np.pi*N)
# 屏幕dpi
dpi=500

axis_x=np.linspace(0,2,num=dpi)
# print(axis_x)

def awgn(x, r):
  r2 = 10 ** (r / 10.0)
  xpower = np.sum(x ** 2) / len(x)
  npower = xpower / r2
  noise = np.random.randn(len(x)) * np.sqrt(npower)
  return x + noise

#频率为FHz,幅度为A的正弦信号
sinx = A*np.sin(1/fs * axis_x)
# sinx=[A*math.sin(i) for i in n]

pl.subplot(221)
pl.plot(axis_x,sinx)
# ts = "fs=%fHz，A=%d的正弦信号" %(fs,A)
ts = "信号"
pl.title(ts,fontproperties='SimHei')
pl.axis('tight')

#频率为FHz、幅值为A的正弦+噪声
gauss = [random.gauss(0,Noise) for i in range(dpi)]
sinAddgua = []
for i in range(len(sinx)):
  sinAddgua.append(sinx[i] + gauss[i])
pl.subplot(222)
pl.plot(axis_x,sinAddgua)
# ts = "fs=%fHz，A=%d的正弦+SNR=%ddb" %(fs,A,Noise)
ts = "信号+SNR=%ddb图" %(Noise)
pl.title(ts,fontproperties='SimHei')
pl.axis('tight')

#正弦信号频谱绘制
Fsin = np.fft.fft(sinx)
Fsin_abs = np.fft.fftshift(abs(Fsin))
axis_xf = np.linspace(-dpi/2,dpi/2-1,num=dpi)
pl.subplot(223)
# ts = "fs=%fHz，A=%d的正弦频谱图" %(fs,A)
ts = "信号频谱图"
pl.title(ts,fontproperties='SimHei')
pl.plot(axis_xf,Fsin_abs)
pl.axis('tight')

#频谱绘制
FsinAddgua = np.fft.fft(sinAddgua)
FsinAddgua_abs = np.fft.fftshift(abs(FsinAddgua))
pl.subplot(224)
# ts = "fs=%fHz，A=%d的正弦+SNR=%ddb频谱图" %(fs,A,Noise)
ts = "信号+SNR=%ddb频谱图" %(Noise)
pl.title(ts,fontproperties='SimHei')
pl.plot(axis_xf,FsinAddgua_abs)
pl.axis('tight')

pl.show()