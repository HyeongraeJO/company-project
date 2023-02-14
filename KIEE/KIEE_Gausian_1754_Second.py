# 그래프 통합

##########      train      ###########
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import numpy

# 데이터는 1754개에서 추출하였음

# dataA = 선간단락 고장 데이터
# dataB = 1선지락 고장 데이터
dataA=pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_type4_6.xls')
dataB=pd.read_excel('/Users/jhr96/Desktop/PYTHON/excel/fault_feature_type1_3.xls')

# x = 선간 단락 전류 크기
x = dataA.filter(['IA_am', 'IA_bm', 'IA_cm','IB_am', 'IB_bm', 'IB_cm'])

x1 = np.array(x).T[0]
x2 = np.array(x).T[1]
x3 = np.array(x).T[2]
x4 = np.array(x).T[3]
x5 = np.array(x).T[4]
x6 = np.array(x).T[5]

current_x = np.concatenate((x1,x2,x3,x4,x5,x6), axis=0)
print('모든 선간 단락 전류 =', current_x)


# y = 1선 지락 전류 크기
y = dataB.filter(['IA_am', 'IA_bm', 'IA_cm','IB_am', 'IB_bm', 'IB_cm'])

y1 = np.array(y).T[0]
y2 = np.array(y).T[1]
y3 = np.array(y).T[2]
y4 = np.array(y).T[3]
y5 = np.array(y).T[4]
y6 = np.array(y).T[5]

current_y = np.concatenate((y1,y2,y3,y4,y5,y6), axis=0)
print('모든 1선 지락 전류 =', current_y)



# 고장 종류 별 전류 평균 
averageA = numpy.mean(current_x)
averageB = numpy.mean(current_y)

print(' 선간 단락 전류 평균 = ', averageA)
print(' 1선 지락 평균  = ', averageB)



# 고장 종류 별 전류 표준편차
stdA = numpy.std(current_x)
stdB = numpy.std(current_y)

print('선간 단락 전류 표준편차 = ', stdA)
print('1선 지락 전류 표준편차 = ', stdB)



# 정규분포 

import scipy.stats


# 선간 단락 전류 정규 분포 
norma_dist = scipy.stats.norm(loc = averageA, scale = stdA)
# 1선 지락 전류 정규 분포 
normb_dist = scipy.stats.norm(loc = averageB, scale = stdB)


# x = 0에서의 확률밀도함수 값 탐색
norma_dist.pdf(0)

# x = 0에서의 확률밀도함수 값 탐색
normb_dist.pdf(0)


# 확률 밀도 함수 그려보고 x = 0인 점에서 값을 표시
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 1000)
density = norma_dist.pdf(x)
plt.plot(x, density)

x = np.linspace(-10, 10, 1000)
density = normb_dist.pdf(x)
plt.plot(x, density)



plt.scatter(0, norma_dist.pdf(0), c = 'r') # x = 0인점 표시
plt.plot([-0.1, 0], [norma_dist.pdf(0), norma_dist.pdf(0)], 'r--')

plt.scatter(0, normb_dist.pdf(0), c = 'r') # x = 0인점 표시
plt.plot([-0.1, 0], [normb_dist.pdf(0), normb_dist.pdf(0)], 'r--')


plt.xlabel('x')
plt.ylabel('density')
plt.xlim(-10, 10)

plt.show()



# 각각 x = 0일때 출력

print (norma_dist.pdf(0))

print (normb_dist.pdf(0))