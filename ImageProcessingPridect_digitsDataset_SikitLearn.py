

# Import the digits dataset 
from sklearn.datasets import load_digits
digits = load_digits()
print (digits.DESCR)
print (digits.target.shape)
print (digits.data.shape)
print (digits.images.shape)


# plot one of the images just for check 
import matplotlib.pyplot as plt
x = digits.images[100]
print(digits.target[100])
plt.imshow(x)
plt.show()


# Split train and test data 
from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test = train_test_split(digits.data , digits.target , test_size = 0.2)


# Pre-processing 
# Normalize the data in a range between (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler(feature_range=(0,1))
x_train  = scaler.fit_transform(x_train)
x_test   = scaler.transform(x_test)
print (x_train[0])

# Define a definition for calculate the metrix
from sklearn.metrics import confusion_matrix , precision_score , recall_score, accuracy_score
def calculate_metrics(y_train,y_test,y_predict_train,y_predict_test):

    accuracy_train   = accuracy_score  (y_true = y_train , y_pred = y_predict_train)
    accuracy_test    = accuracy_score  (y_true = y_test  , y_pred = y_predict_test )
    confusion_test   = confusion_matrix(y_true = y_test  , y_pred = y_predict_test )
    precision_test   = precision_score (y_true = y_test  , y_pred = y_predict_test , average="weighted")
    recall_test      = recall_score    (y_true = y_test  , y_pred = y_predict_test , average="weighted")

    print ("confusion      :\n" , confusion_test*100 ,
        "\n precision      : "  , precision_test*100 ,"%",
        "\n recall         : "  , recall_test   *100 ,"%",
        "\n accuracy_Train : "  , accuracy_train*100 ,"%",
        "\n accuracy_Test  : "  , accuracy_test *100 ,"%",
        )
    return accuracy_train,accuracy_test,confusion_test,precision_test,recall_test


# Mode selection
# First Model (Random_Forest)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=128 , n_estimators=256)
rf.fit(x_train, y_train)

#Prediction 
y_predict_train = rf.predict(x_train)
y_predict_test  = rf.predict(x_test)
accuracy_train_rf,accuracy_test_rf,confusion_test_rf,precision_test_rf,recall_test_rf = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# Second Model (SVC)
from sklearn.svm import SVC
svm = SVC(kernel="linear")
svm.fit(x_train,y_train)

#Prediction 
y_predict_train = svm.predict(x_train)
y_predict_test  = svm.predict(x_test)
accuracy_train_svm,accuracy_test_svm,confusion_test_svm,precision_test_svm,recall_test_svm = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# Fourth Model (Nural_Network)
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=256 , batch_size= 64 , solver= "lbfgs" , learning_rate="adaptive")
ann.fit(x_train,y_train)

#Prediction 
y_predict_train = ann.predict(x_train)
y_predict_test  = ann.predict(x_test)
accuracy_train_ann,accuracy_test_ann,confusion_test_ann,precision_test_ann,recall_test_ann = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# Fifth Model (K_Neighbors)
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)

#Prediction 
y_predict_train = knn.predict(x_train)
y_predict_test  = knn.predict(x_test)
accuracy_train_knn,accuracy_test_knn,confusion_test_knn,precision_test_knn,recall_test_knn = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)


# Plot the metrix and compare them 
# Plot of Trained data Accuracy
import matplotlib.pyplot as plt
Accu_train = [accuracy_train_knn,accuracy_train_rf,accuracy_train_svm,accuracy_train_ann]
tittle     = ["KNN" , "RF" , "SVM", "ANN"]
colors     = ["Blue", "Red", "Yellow" , "orange"]
plt.bar(tittle,Accu_train , color = colors)
plt.title("Acuuracy in Trained Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Accuracy")
plt.show()

# Plot of Test data Accuracy
import matplotlib.pyplot as plt
Accu_test = [accuracy_test_knn,accuracy_test_rf,accuracy_test_svm,accuracy_test_ann]
tittle     = ["KNN" , "RF" , "SVM", "ANN"]
colors     = ["Blue", "Red", "Yellow" , "orange"]
plt.bar(tittle, Accu_test , color = colors)
plt.title("Acuuracy in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Accuracy")
plt.show()

# Plot of Precition 
import matplotlib.pyplot as plt
Precition = [precision_test_knn,precision_test_rf,precision_test_svm,precision_test_ann]
tittle     = ["KNN" , "RF" , "SVM", "ANN"]
colors     = ["Blue", "Red", "Yellow" , "orange"]
plt.bar(tittle, Precition , color = colors)
plt.title("Precition in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Precition")
plt.show()

# Plot of Recall
import matplotlib.pyplot as plt
recall = [recall_test_knn,recall_test_rf,recall_test_svm,recall_test_ann]
tittle     = ["KNN" , "RF" , "SVM", "ANN"]
colors     = ["Blue", "Red", "Yellow" , "orange"]
plt.bar(tittle, recall , color = colors)
plt.title("Recall in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Recall")
plt.show()
