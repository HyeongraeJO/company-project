#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/jhr96/Desktop/PYTHON/excel/train 1L 1000.xlsx')

X=data.drop(['target','type'],axis=1)

X_c=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph','VC_cm','VC_cph','IC_cm','IC_cph','VD_cm','VD_cph','ID_cm','ID_cph'])

y=data.filter(['target'])
z=data.filter(['type'])

row=y.shape[0]
yy=y.to_numpy()
z=z.to_numpy()



# one-hot     
for i in range(0,row):
    if z[i]==4: # ab상 정상 0
        yy[i]=0
    elif z[i]==5: # bc상 정상 
        yy[i]=0
    elif z[i]==6: # ca상 정상 0
        yy[i]=yy[i]
    else:         # 정상데이터 있을 시에 
        yy[i]=0


yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)

print('yy')
print(yy)



#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
lda.fit(X_c,y)
#lda.fit(X_b,y)
#lda.fit(X_c,y)

Xc_lda=lda.transform(X_c)
#Xb_lda=lda.transform(X_b)
#Xc_lda=lda.transform(X_c)

###############################
print( '\nLDA 적용 후 데이터 셋')
lda_columns = ['lda_comp1', 'lda_comp2', 'lda_comp3']


Xc_lda_df = pd.DataFrame(Xc_lda, columns = lda_columns)

Xc_lda_df['target'] = y

print( '\n head(5) 출력 결과 : ')
print(Xc_lda_df.head(5))

# 3차원!!!!
# 3차원 그래프세팅
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')

# LDA 시각화
import matplotlib.pyplot as plt
df = Xc_lda_df
markers = ['o','x','^','h','D','s','.',',','v','<','>','1']
labels = ['Sect1', 'Sect2', 'Sect3', 'Sect4', 'Sect5', 'Sect6', 'Sect7', 'Sect8', 'Sect9', 'Sect10', 'Sect11', 'Sect12']

for i, mark in enumerate(markers):
    X_i = df[df['target']== i+1]
    target_i = labels[i]
    X1 = X_i['lda_comp1']
    X2 = X_i['lda_comp2']
    X3 = X_i['lda_comp3']
    ax.scatter(X1, X2, X3 , label=target_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
ax.legend(bbox_to_anchor=(1.25, 1))
plt.show()
