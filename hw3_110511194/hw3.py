from PIL import Image
import numpy as np
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

matrix_list = []

for i in range(5000):
    filename = f"picture/train/train_{i+1}.png"
    img = Image.open(filename).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_vector = np.reshape(img_array, (1, 28*28))
    img_vector = img_array.flatten()
    matrix_list.append(img_vector)

X_train = np.array(matrix_list)

labels_list = [i for i in range(5) for j in range(1000)]
y_train = np.array(labels_list)

matrix_list = []

for i in range(2500):
    filename = f"picture/test/test_{i+1}.png"
    img = Image.open(filename).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_vector = np.reshape(img_array, (1, 28*28))
    img_vector = img_array.flatten()
    matrix_list.append(img_vector)

X_test = np.array(matrix_list)

labels_list = [i for i in range(5) for j in range(500)]
y_test = np.array(labels_list)

nu_values = [0.1, 0.3, 0.5, 0.99]
c_values = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
kernel_types = ['linear', 'rbf', 'sigmoid']

for nu in nu_values:
    for kernel in kernel_types:
        clf = NuSVC(nu=nu, kernel=kernel)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(f"Accuracy for nu={nu}, kernel={kernel}: {accuracy}")

for c in c_values:
    for kernel in kernel_types:
        clf = SVC(C=c, kernel=kernel)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(f"Accuracy for C={c}, kernel={kernel}: {accuracy}")


clf = SVC(C=3.0, kernel='rbf')
clf.fit(X_train, y_train)
support_vectors = clf.support_vectors_

for i in range(support_vectors.shape[0]):
    image_matrix = support_vectors[i].reshape(28, 28)
    image = Image.fromarray(image_matrix).convert('L')
    image.save(f'support_vectors/support_vector_{i+1}.png')
