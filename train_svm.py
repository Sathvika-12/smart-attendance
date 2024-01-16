import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

features = np.load('cnn_features.npy')
labels = np.array([i // 100 for i in range(features.shape[0])])  # Assuming 100 images per class

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy * 100:.2f}%')

# Save the SVM model
import joblib
joblib.dump(svm_model, 'svm_model.joblib')
