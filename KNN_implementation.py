import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

column_names = ["target_vector", "region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", "short-line-density-2", "vedge-mean", "vedge-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean"]
df = pd.read_csv("segmentation_new.csv", header=None, skiprows=4, names=column_names)
df.head(10)

df['target_vector'] = df['target_vector'].map({'GRASS': 0, 'PATH': 1, 'WINDOW' : 2, 'CEMENT' : 3, 'FOLIAGE' : 4, 'SKY' : 5, 'BRICKFACE' : 6})
df.dropna(inplace=True)
df.head(10)


class_groups = [
    (0, 1), 
    (2, 3),
    (4, 5),
    (6)
]

class_colors = {
    0: "blue",
    1: "red",
    2: "green",
    3: "yellow",
    4: "grey",
    5: "pink",
    6: "indigo"
}

class_labels = {
    0: "GRASS",
    1: "PATH",
    2: "WINDOW",
    3: "CEMENT",
    4: "FOLIAGE",
    5: "SKY",
    6: "BRICKFACE"
}


for label in df.columns[1:]: 
    min_val = df[label].min()
    max_val = df[label].max()
    print(f"Column: {label}, Min: {min_val}, Max: {max_val}")
    bins = np.linspace(min_val, max_val, 15)
    
    plt.figure()
    for cls in class_groups[0]:
        plt.hist(df[df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 1")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
    
    plt.figure()
    for cls in class_groups[1]:
        plt.hist(df[df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 2")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

    plt.figure()
    for cls in class_groups[2]:
        plt.hist(df[df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 3")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(df[df["target_vector"] == 6][label], bins=bins, color=class_colors[6], label=class_labels[6], alpha=0.7, density=True)
    plt.title(f"{label} - Group 4")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


len(df[df["target_vector"] == 0]), len(df[df["target_vector"] == 1]), len(df[df["target_vector"] == 2]), len(df[df["target_vector"] == 3]), len(df[df["target_vector"] == 4]), len(df[df["target_vector"] == 5]), len(df[df["target_vector"] == 6])

X = df[["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", "short-line-density-2", "vedge-mean", "vedge-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean"]].values
y = df["target_vector"].values
print(X.shape, y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.hstack((X, np.reshape(y, (-1, 1))))
transformed_df = pd.DataFrame(data, columns = ["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", "short-line-density-2", "vedge-mean", "vedge-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean", "target_vector"])
transformed_df.head(10)

len(transformed_df[transformed_df["target_vector"] == 0]), len(transformed_df[transformed_df["target_vector"] == 1]), len(transformed_df[transformed_df["target_vector"] == 2]), len(transformed_df[transformed_df["target_vector"] == 3]), len(transformed_df[transformed_df["target_vector"] == 4]), len(transformed_df[transformed_df["target_vector"] == 5]), len(transformed_df[transformed_df["target_vector"] == 6])

class_groups = [
    (0, 1), 
    (2, 3),
    (4, 5),
    (6)
]

class_colors = {
    0: "blue",
    1: "red",
    2: "green",
    3: "yellow",
    4: "grey",
    5: "pink",
    6: "indigo"
}

class_labels = {
    0: "GRASS",
    1: "PATH",
    2: "WINDOW",
    3: "CEMENT",
    4: "FOLIAGE",
    5: "SKY",
    6: "BRICKFACE"
}


for label in df.columns[1:]: 
    min_val = transformed_df[label].min()
    max_val = transformed_df[label].max()
    print(f"Column: {label}, Min: {min_val}, Max: {max_val}")
    bins = np.linspace(min_val, max_val, 15)
    
    plt.figure()
    for cls in class_groups[0]:
        plt.hist(transformed_df[transformed_df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 1")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
    
    plt.figure()
    for cls in class_groups[1]:
        plt.hist(transformed_df[transformed_df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 2")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

    plt.figure()
    for cls in class_groups[2]:
        plt.hist(transformed_df[transformed_df["target_vector"] == cls][label], bins=bins, color=class_colors[cls], label=class_labels[cls], alpha=0.7, density=True)
    plt.title(f"{label} - Group 3")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(transformed_df[transformed_df["target_vector"] == 6][label], bins=bins, color=class_colors[6], label=class_labels[6], alpha=0.7, density=True)
    plt.title(f"{label} - Group 4")
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


X_columns = df[["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", "short-line-density-2", "vedge-mean", "vedge-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean"]]
y_columns = df["target_vector"]
X_train, X_test, y_train, y_test = train_test_split(X_columns, y_columns, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

X_columns_transformed = transformed_df[["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5", "short-line-density-2", "vedge-mean", "vedge-sd", "hedge-mean", "hedge-sd", "intensity-mean", "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean", "value-mean", "saturation-mean", "hue-mean"]]
y_columns_transformed = transformed_df["target_vector"]
X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = train_test_split(X_columns_transformed, y_columns_transformed, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train_transformed, y_train_transformed)
y_pred_transformed = knn_model.predict(X_test_transformed)
print(classification_report(y_test_transformed, y_pred_transformed))







