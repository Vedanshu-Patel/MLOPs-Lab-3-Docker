import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
if __name__ == '__main__': 
    # Load the Iris dataset
    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target
    wine_dataset = datasets.load_wine()
    X, y = wine_dataset.data, wine_dataset.target
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Build a simple TensorFlow model
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(13,), activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=120, validation_data=(X_test, y_test))

    model.save('my_model.keras')
    pickle.dump(sc, open('scaler.pkl', 'wb'))
    print("Model was trained and saved")
