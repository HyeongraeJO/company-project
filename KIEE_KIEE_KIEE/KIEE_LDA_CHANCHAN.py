#LDA
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_excel('C:/Users/CHOI/Desktop/train_1L_1000.xlsx') # ab외 정상 데이터 


#X_a=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
#X_b=data.filter(['VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph'])
#X_c=data.filter(['VA_cm','VA_cph','IA_cm','IA_cph','VB_cm','VB_cph','IB_cm','IB_cph'])


'''
1.선간 또한 ab, bc, ca 데이터를 따로 보아야할 이유는 ?
2.ab 선간 단락시 ab의 둘다 특성이 뚜렷하니 ab의 데이터를 모두 가져 오는 것이 맞는가?
'''

#X_ab=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph'])
X_ab=data.filter(['VA_am','VA_aph','IA_am','IA_aph','VB_am','VB_aph','IB_am','IB_aph','VA_bm','VA_bph','IA_bm','IA_bph','VB_bm','VB_bph','IB_bm','IB_bph','VC_am','VC_aph','IC_am','IC_aph','VC_am','VC_aph','IC_am','IC_aph','VC_bm','VC_bph','IC_bm','IC_bph','VC_bm','VC_bph','IC_bm','IC_bph','VD_am','VD_aph','ID_am','ID_aph','VD_am','VD_aph','ID_am','ID_aph','VD_bm','VD_bph','ID_bm','ID_bph','VD_bm','VD_bph','ID_bm','ID_bph'])

y=data.filter(['target'])
z=data.filter(['type'])



row=y.shape[0]
yy=y.to_numpy()

z=z.to_numpy()

    
# one-hot     
for i in range(0,row):
    if z[i]==1: # a상
        yy[i]=0#a상 
    elif z[i]==2: # b상 정상 0
        yy[i]=0
    elif z[i]==3: # c상 정상 0
        yy[i]=yy[i]


yy=pd.DataFrame(yy)
z_1=pd.DataFrame(z)
print('target를출력하라')
print(yy)

print('type를출력하라')
print(z_1)
#tn_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
#lda.fit(X_a,y)
#lda.fit(X_b,y)
#lda.fit(X_c,y)
lda.fit(X_ab,y)

#Xa_lda=lda.transform(X_a)
#Xb_lda=lda.transform(X_b)
#Xc_lda=lda.transform(X_c)
Xab_lda=lda.transform(X_ab)

#print(Xa_lda.shape)
#print(Xb_lda.shape)
#print(Xc_lda.shape)
print(Xab_lda.shape)

print(lda.explained_variance_ratio_)

print(np.sum(lda.explained_variance_ratio_[0:10]))


lda_columns=['lda_comp1','lda_comp2','lda_comp3']
Xab_lda_df=pd.DataFrame(Xab_lda,columns=lda_columns)
Xab_lda_df['target']=y
Xab_lda_df.head(11)



df = Xab_lda_df
print(df)

markers=['o','.','X','s','D','v','^','*','s','x','_','2','1'] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, mark in enumerate(markers):
    df_i = df[df['target']== i]
    target_i = i
    X1 = df_i['lda_comp1']
    X2 = df_i['lda_comp2']
    X3 = df_i['lda_comp3']
    ax.scatter(X1, X2, X3,
                marker=mark, 
                label=target_i)

ax.set_xlabel('lda_component1')
ax.set_ylabel('lda_component2')
ax.set_zlabel('lda_component3')
ax.legend(loc='best')
plt.savefig('test.png')
plt.show()