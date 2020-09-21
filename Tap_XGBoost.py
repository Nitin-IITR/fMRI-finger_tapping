import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tap_series_all22=pd.DataFrame(tap_series_all22)
rest_series_all22=pd.DataFrame(rest_series_all22)


df = tap_series_all22.append(rest_series_all22)
df= df.reset_index(drop=True)

    

X= df.iloc[:,:].values

################################################
######### Nor 23.6 sec

Y = np.concatenate((np.zeros(len(tap_series_all22)), np.ones(len(rest_series_all22))))
#Y = np.concatenate((np.zeros(shape=855), np.ones(shape=855)))

############################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



acs=[]

for i in range(20):
    from sklearn.model_selection import train_test_split
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.25)
    
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train,Y_train)

    Y_pred = classifier.predict(X_test)

    
#    from sklearn.metrics import confusion_matrix
#    cmf= confusion_matrix(Y_test,Y_pred)
    
    from sklearn.metrics import accuracy_score
    acs.append(accuracy_score(Y_test,Y_pred))


acs1 =np.mean(acs) 
acs1=acs1*100

std=np.std(acs)
std=std*100

acs1

################################################
####### MUCH BETTER THAN BELOW (CHECKED BY ME)
# plot feature importance
from xgboost import plot_importance
plot_importance(classifier)
plt.show()


gain=classifier.get_booster().get_score(importance_type="gain")

def getList(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key) 
          
    return list

keys=getList(gain)

x = [key.split("f") for key in keys]

feature_imp=[]
for i in range(len(x)):
    feature_imp.append(int(x[i][1]))

np.transpose(feature_imp)

################################################
####### LESS RELIABLE 
# plot
print(classifier.feature_importances_)
plt.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
plt.show()

feature_imp_frame= np.transpose(pd.DataFrame([classifier.feature_importances_ ,np.linspace(0,len(classifier.feature_importances_)-1,len(classifier.feature_importances_))]))
feature_imp_sorted= acs_frame.sort_values(by=0, ascending=False)

A=feature_imp_sorted.iloc[:,1].values
A.astype(int)
