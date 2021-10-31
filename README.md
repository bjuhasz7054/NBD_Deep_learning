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
