## CNN

MLPs only take vectors as input. Convert matrices to vector. After encoding our images as vectors, they can by fed into the input layer of an MLP.
we'll create a neural network for discovering the patterns in our data.
Add a softmax activation function to the final fully connected layer. It ensures that the network outputs an estimate for the probability that each potential digit is depicted in the image.

```python
from keras.models import Sequential
from keras.layer import Dense, Flatten

#define the model 
model = Sequential()
# Flatten layer, it takes the image matrix's input and converts it to a vector 
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='ReLU'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='ReLU'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.
#summarize the model
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model 
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)
# load the weights that yielded the best validation accuracy
model.load_weights('mnist.model.best.hdf5')
# evaluate test accuracy 
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]
print('Test accuracy: %.4f%%' % accuracy)
```

* ReLU activation function leaves positive values alone and sends all negative value to zero. It helps with what's known as the vanishing gradients problem. By adding the ReLU funtion, our model is able to attain much better accuracy. The activation function will also be extensively used in convolutional neural networks. 
* In order to minimize over-fitting, we can add dropout layers. The dropout layers must be supplied a parameter between zero and one. Thicorresponds to the probability that any node in the network is removed during training. When deciding its value, it's recommeded to start small and see how the network responds.
* Loss function. Since we're constructing a multiclass classifier, we'll use categorical cross-entropy loss. This loss function checks to see if our model has done a good job with classifying an image by comparing the models prediction to the true label.
1. Load MNIST Database
2. Visualize the fix six training images
3. View an image in more detail
4. Rescale the images by dividing every pixel in every image by 255
5. Encode categorical integer labels using a one-hot scheme
6. Define the model Architecture
7. Compile the model
8. Calculate the classification Accuracy on the Test Set
9. Train the model
10. Load the model with the best classification accuracy on the validation set
11. Calcualate the classfication accuracy on the test set

We tell the network explicitly that objects and images are largely the same whether they're on the left or the right of the picture. This is partially accomplished through weight sharing. All of these ideas will motivate so-called convolutional layers. We motivated the idea of replacing a fully connected layer with a locally connected layer. This locally connected layer contained fewer weights, which were shared across space. 
Towards breaking up the image for building a convolutional layer, we first select a width and height that defines a convolution window. We then simply slide this window horizontally and vertically over the matrix of image pixels.


