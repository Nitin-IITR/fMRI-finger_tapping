import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#############################
tap_series_all22=pd.DataFrame(tap_series_all22)
rest_series_all22=pd.DataFrame(rest_series_all22)

df = tap_series_all22.append(rest_series_all22)
df= df.reset_index(drop=True)


acs_column=[]
std_column=[]
    
for k in range(len(tap_series_all22.columns)):
    
    #X= df.iloc[:,[0., 1., 2., 3., 4., 5., 6., 7., 8.,36., 37., 38., 39., 40., 41., 42., 43., 44.]].values
    
    X= df.iloc[:,k].values

    ################################################
    ######### Nor 23.6 sec
    
    Y = np.concatenate((np.zeros(len(tap_series_all22)), np.ones(len(rest_series_all22))))
    
    #Y = np.concatenate((np.zeros(shape=855), np.ones(shape=855)))
    
    ############################################################
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X.reshape((X.shape[0], 1)))
    
    
    
    acs=[]
    
    for i in range(20):
        from sklearn.model_selection import train_test_split
        X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.25)
        
        from sklearn.svm import SVC
        svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
        svc_reg.fit(X_train,Y_train)
    
        Y_pred = svc_reg.predict(X_test)
    
        
    #    from sklearn.metrics import confusion_matrix
    #    cmf= confusion_matrix(Y_test,Y_pred)
        
        from sklearn.metrics import accuracy_score
        acs.append(accuracy_score(Y_test,Y_pred))
    
    
    acs1 =np.mean(acs) 
    acs1=acs1*100
    
    std=np.std(acs)
    std=std*100
    
    acs_column.append(acs1)
    std_column.append(std)
    
    
  
######################################################################################   
######################################################################################    
    
# Now selecting on the features with top accuracy and checking there combined accuracy


acs_frame= np.transpose(pd.DataFrame([acs_column ,np.linspace(0,len(acs_column)-1,len(acs_column))]))
acs_sorted= acs_frame.sort_values(by=0, ascending=False)

Index= acs_sorted.iloc[:,1].values
Index= Index[0:2]


tap_series_all22[Index]

df = tap_series_all22[Index].append(rest_series_all22[Index])
df= df.reset_index(drop=True)

    

X= df.iloc[:,:].values

################################################
######### Nor 23.6 sec

Y = np.concatenate((np.zeros(len(tap_series_all22)), np.ones(len(rest_series_all22))))
#Y = np.concatenate((np.zeros(shape=855), np.ones(shape=855)))

############################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



acs=[]

for i in range(20):
    from sklearn.model_selection import train_test_split
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.25)
    
    from sklearn.svm import SVC
    svc_reg = SVC(kernel ='rbf',C=1,gamma='auto')
    svc_reg.fit(X_train,Y_train)

    Y_pred = svc_reg.predict(X_test)

    
#    from sklearn.metrics import confusion_matrix
#    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))


acs1 =np.mean(acs) 
acs1=acs1*100

std=np.std(acs)
std=std*100








