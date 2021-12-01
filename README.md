# NBD Deep Learning: Recognizing facial features
Members of the team: Domonkos Debreczeni, Nándor Szécsényi, Benedek Juhász

## Aims and goals of the project.

The topic of the project is the 3rd from the published list, "Recognizing facial features", and all of our goals and ideas are inspired by the topic's description.

We chose an approach new to us, involving working in the latent space. Let us clarify what we mean by looking at a possible schedule for this semester:
1. exploratory data analysis and data preparation
2. training a multi-output/multilabel neural network on the FairFace dataset
3. analyze latent space vectors of the trained model given specific inputs
4. Using the knowledge acquired in the last phase, we can continue our project in many ways:
- modification of latent space vectors, which can then be fed to a decoding subnetwork (resulting in a change in the input face's features)
- creating latent space vectors by "hand" and then examining the decoded faces.

We want to emphasize that the former schedule is merely a plan; we are open to any ideas and experiences during this project.

## Milestone no. 2
### Progression
For the second milestone, we experimented with multiple models and training techniques. Our progression so far in broad strokes:
-	we preprocessed and loaded the training data using the ImageDatagenerator class and its flow_from_dataframe method, which suits our label files perfectly
-	using a VGG network that was pre-trained on faces, we used transfer learning to create our models and began training them
-	we evaluated our models with multiple metrics, including precision, recall and for better visual understandability, confusion matrices
-	an AutoEncoder network has been implemented (using the trained classifier network as the encoder) to further analyze the properties of our dataset and the transmission between two classes in the latent space

It is worth mentioning that until now, two model architectures have achieved the best classification results. The two models differ from each other only in their final layers:
-	the first is a multilabel model that has one output layer, producing an 18x1 vector
-	the second is a multi-output model that has three output layers for the three different categories(race, age, gender)

Currently, we are working with the latter because it achieved a better accuracy overall, although the prediction of the multilabel model can be configured to achieve better results.

### Running the code
After opening the notebook in Colab, run all the cells, and in due time the trained model will be evaluated after the training. The hyperparameters in one of the first cells can be tuned. Running with the current config trains the network on a fraction of the data for 30 epochs. We chose this setup because it is quicker for a demonstration, and it is quicker to try out a change in the model architecture.

### Plans for the future
- Hyperparameter optimization (focusing on the number of layers to train and activation functions)
- Analyzing the latent space of the model and identifying learned features
- Using further techniques to increase accuracies (dropout, attention)
- Fix multilabel prediction to be able to compare the two model architectures properly
