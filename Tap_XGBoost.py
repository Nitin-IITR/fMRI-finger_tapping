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



65,  1,  3,  0, 66,  2, 31, 20, 25,  5,  9, 16, 12, 59, 19,  6, 26,
       48, 22, 56, 52, 28, 30, 51, 50, 53, 61, 35, 64,  7, 60, 34, 36, 27,
       17, 10, 55, 15,  8, 46,  4, 63, 18, 62, 32, 13, 44, 29, 40, 21, 42,
       38, 23, 49, 39, 58, 54, 45, 33, 57, 47, 24, 41, 37, 11, 43, 14

65,  1, 34, 16,  0, 20,  2, 66,  3, 25, 36,  7, 59, 26, 15, 35, 17,
        5,  6, 51, 19, 47, 31, 48, 33, 56, 53, 63, 60, 21, 61, 27, 44, 13,
       22, 49, 43, 10, 52, 55, 28, 64, 62,  8, 18, 50, 32, 11, 42, 54, 57,
       58, 24,  9, 12, 23,  4, 46, 29, 30, 39, 14, 37, 40, 38, 45





