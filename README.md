# NBD_Deep_learning: Recognizing facial features
Members of the team: Domonkos Debreczeni, Nándor Szécsényi, Benedek Juhász


Aims and goals of our project.

Just to be clear, our project's topic is the 3rd from the list, "Recognizing facial features" and all of our goals and ideas are inspired by the topic's official description. 

Given our team's average experience, in this semester we chose a (for us) new approach, which involves working in the latent space. Let us clarify what we mean by looking at a possible schedule for this semester:
1. First (after all the data preparation and exploration in this notebook) we train a neural network on the multi label classification dataset, FairFace.
2. After training the model and reaching a high enough accuracy (its exact quantity hasn't been determined), we will begin analyzing the latent spaces vectors given for specific inputs.
3. Using the knowledge acquired in the last phase, we can continue our project in many ways:
- modification of latent space vectors, which can then be fed to a decoding subnetwork (resulting in a change in the input face's features)
- creating latent space vectors by "hand" and then examining the decoded faces.

We would like to emphasize that, the former schedule is merely a plan, we are open to any ideas and experiences during this project. 

##Milestone no. 2
###Progression
For the second milestone we experimented with multiple models and training techniques. Our progression so far in broad strokes:
-	we preprocessed and loaded the training data using the ImageDatagenerator class, and its flow_from_dataframe method, which suits our label files perfectly
-	using a pretrained VGG network on faces and transfer learning, we created our own models and began training them
-	we evaluated our models with multiple metrics, including for instance precision, recall and the better visual understandability, confusion matrices
-	also an AutoEncoder network has been implemented (using the trained classifier network as the encoder), to further analyze the properties of our dataset and the transmission between two classes in the latent space
Worth mentioning, that right up until now, two model architectures had achieved the best classification results. The two models differ from each other only in their final layers:
-	the first one had only one final layer, producing a 9x1 vector
-	the other had 3 final layers as there are 3 different categories(race, age, gender)
Currently we are working the latter, since the first one’s prediction was off due to its output.
###Running the code
After opening the notebook in Colab, just run all the cells and in due time the trained model will be evaluated after the training. The hyperparameters in the first couple of cells can be modified, running with the current setup would train the network on only a fraction of the data for 30 epochs. We chose this setup, since it’s quicker for a demonstration in case you want to run it and sometimes it’s easier for us to see if a model architecture is able to learn or not.
###Plans in the future
-	Hyperparameter optimization (focusing on number of layers to train and activation functions)
-	Analyzing the latent space of the model and identifying learned features
-	Using further techniques to increase accuracies (dropout, attention)
