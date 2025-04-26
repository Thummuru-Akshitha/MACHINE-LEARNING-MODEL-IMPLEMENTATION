# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: THUMMURU AKSHITHA

*INTERN ID*: CT06WR252

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

*DESCRIPTION OF MACHINE-LEARNING-MODEL-IMPLEMENTATION*:

This Python program is a simple yet powerful example of a machine learning-based spam detection system using the Naive Bayes algorithm. It uses a small sample dataset consisting of text messages, where each message is labeled either as spam (1) or not spam (0). The main goal of the program is to train a classifier that can predict whether a new, unseen message is spam. The workflow begins by splitting the available data into a training set and a testing set using Scikit-learn’s **train_test_split** function, ensuring that 30% of the data is reserved for evaluating the model's performance. Once the data is split, the next important step is text preprocessing. Since machine learning models cannot work directly with raw text, the program converts the text into numerical representations using **CountVectorizer**, which transforms the text into a matrix of token counts. This numeric format captures the frequency of words across all messages, enabling the model to learn patterns.

After preprocessing, the model training phase begins by using a **Multinomial Naive Bayes (MultinomialNB)** classifier. Naive Bayes models are particularly well-suited for text classification problems, especially when dealing with word counts or term frequencies. The model is trained on the vectorized training messages and their corresponding labels. Once trained, the model is used to predict the labels for the test set. The program then evaluates the model’s performance by calculating the **accuracy score**, which measures the percentage of correctly classified messages. It also prints a **classification report** that provides detailed metrics like precision, recall, and F1-score for both spam and non-spam categories, helping users understand how well the model distinguishes between the two classes.

One of the interactive features of the program is that it allows the user to input their own custom message to see if the model identifies it as spam or not. The user's message is vectorized using the same **CountVectorizer** used during training to ensure consistency, and then the model makes a prediction based on this input. The prediction is then displayed as either "Spam" or "Not Spam," giving users a real-time feel of how spam detection models work. This small but complete project covers several important concepts in machine learning and natural language processing (NLP), including text preprocessing, model training, evaluation, and deployment in a user-friendly interface.

Overall, this program serves as an excellent starting point for beginners looking to understand how to build a text classification model from scratch. It demonstrates the importance of converting unstructured text into structured numerical data, selecting the right machine learning algorithm, and properly evaluating model performance. Although the sample dataset used here is small and simple, the same techniques can easily scale up to handle much larger datasets and more complex classification tasks, such as building real-world email spam filters. By tweaking and extending this project—such as by adding more training data, experimenting with different vectorization techniques like TF-IDF, or using more advanced algorithms—learners can further deepen their understanding of machine learning and NLP in Python.
