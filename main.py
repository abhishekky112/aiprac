# Practical 1
# Breath first search
# Depth first search

class BFS:
    def __init__(self, graph, start, goal):
        self.start = start
        self.goal = goal
        self.graph = graph

    def bfs_shortest_path(self):
        explored = []
        queue = [[self.start]]
        
        if self.start == self.goal:
            return "That was easy! Start = goal"
        
        while queue:
            path = queue.pop(0)  # Dequeue the first path
            node = path[-1]      # Get the last node from the path
            
            if node not in explored:
                neighbours = self.graph[node]
                
                for neighbour in neighbours:
                    new_path = list(path)  # Create a new path
                    new_path.append(neighbour)  # Add the neighbour to the path
                    queue.append(new_path)  # Enqueue the new path
                    
                    if neighbour == self.goal:
                        return new_path  # Return the path if the goal is found
                
                explored.append(node)  # Mark the node as explored
        
        return "So sorry, but a connecting path doesn't exist :("

# Graph definition
graph = {
    'A': ['Z', 'S', 'T'],
    'B': ['U', 'P', 'G', 'F'],
    'C': ['D', 'R', 'P'],
    'D': ['M'],
    'E': ['H'],
    'I': ['V', 'N'],
    'L': ['T', 'M'],
    'O': ['Z', 'S'],
    'P': ['R'],
    'U': ['V'],
    'Z': ['O', 'A'],
    'S': ['O', 'A', 'R', 'F'],
    'T': ['A', 'L'],
    'M': ['L', 'D'],
    'R': ['S', 'P', 'C'],
    'F': ['S', 'B']
}

# Create an instance of the BFS class and find the path
bfs_instance = BFS(graph, 'A', 'B')
print("BFS Path:", bfs_instance.bfs_shortest_path())

graph = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Bucharest': ['Urziceni', 'Pitesti', 'Giurgiu', 'Fagaras'],
    'Craiova': ['Dobreta', 'Rimnicu Vilcea', 'Pitesti'],
    'Dobreta': ['Mehadia'],
    'Eforie': ['Hirsova'],
    'Iasai': ['Vaslui', 'Neamt'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Pitesti': ['Rimnicu Vilcea'],
    'Urziceni': ['Vaslui'],
    'Zerind': ['Oradea', 'Arad'],
    'Sibiu': ['Oradea', 'Arad', 'Rimnicu Vilcea', 'Fagaras'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Mehadia': ['Lugoj', 'Dobreta'],
    'Rimnicu Vilcea': ['Sibiu', 'Pitesti', 'Craiova'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Giurgiu': ['Bucharest'],
    'Vaslui': ['Urziceni', 'Iasai'],
    'Neamt': ['Iasai']
}

def IDDFS(root, goal):
    depth = 0
    while True:
        print(f"Looping at depth {depth}")
        result = DLS(root, goal, depth)
        if result == goal:
            return result
        depth += 1

def DLS(node, goal, depth):
    if depth == 0 and node == goal:
        return node
    elif depth > 0:
        for child in graph.get(node, []):
            if goal == DLS(child, goal, depth - 1):
                return goal
    return None

# Running the IDDFS
result = IDDFS('Arad', 'Bucharest')
print("IDDFS Result:", result)



# Practical 2 
# Implement A* search algorithm for Romanian map problem.
import queue as Q

# Romanian Map Problem data (graph and heuristic)
dict_gn = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Pitesti': 97, 'Craiova': 146},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Bucharest': 101},
    'Craiova': {'Rimnicu Vilcea': 146, 'Drobeta': 120},
    'Drobeta': {'Craiova': 120, 'Mehadia': 75},
    'Mehadia': {'Drobeta': 75, 'Lugoj': 70},
    'Lugoj': {'Mehadia': 70, 'Timisoara': 118},
    'Timisoara': {'Lugoj': 118, 'Arad': 118},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101},
}

dict_hn = {
    'Arad': 366,
    'Zerind': 374,
    'Oradea': 380,
    'Sibiu': 253,
    'Fagaras': 178,
    'Rimnicu Vilcea': 193,
    'Pitesti': 98,
    'Craiova': 160,
    'Drobeta': 242,
    'Mehadia': 241,
    'Lugoj': 226,
    'Timisoara': 199,
    'Bucharest': 0,
}

start = 'Arad'
goal = 'Bucharest'
result = ''

def get_fn(citystr):
    cities = citystr.split(" , ")
    gn = 0
    for ctr in range(0, len(cities) - 1):
        gn += dict_gn[cities[ctr]][cities[ctr + 1]]
    hn = dict_hn[cities[-1]]
    return hn + gn

def expand(cityq):
    global result
    tot, citystr, thiscity = cityq.get()
    if thiscity == goal:
        result = citystr + " : : " + str(tot)
        return
    for cty in dict_gn[thiscity]:
        cityq.put((get_fn(citystr + " , " + cty), citystr + " , " + cty, cty))
    expand(cityq)

def main():
    cityq = Q.PriorityQueue()
    cityq.put((get_fn(start), start, start))
    expand(cityq)
    print("The A* path with the total cost is:")
    print(result)

main()


from queue import PriorityQueue

v = 14
graph = [[] for _ in range(v)]

# Function for implementing Best First Search
def best_first_search(actual_Src, target, n):
    visited = [False] * n
    pq = PriorityQueue()
    pq.put((0, actual_Src))
    visited[actual_Src] = True

    while not pq.empty():
        u = pq.get()[1]
        print(u, end=" ")
        if u == target:
            break

        for v, c in graph[u]:
            if not visited[v]:
                visited[v] = True
                pq.put((c, v))
    print()

# Function for adding edges to the graph
def addedge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))

# Adding edges (graph representation)
addedge(0, 1, 3)  # Example graph
addedge(0, 2, 6)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 12)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 2)

source = 0  # Starting node (as an example)
target = 9  # Target node (as an example)
best_first_search(source, target, v)



# Practical 3: Decision Tree Learning
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

# Create a sample dataset
data = {
    'outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 
                'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild', 
             'Mild', 'Mild', 'Hot', 'Mild'],
    'humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'High'],
    'windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True', 'False', 'False', 'False', 
              'True', 'True', 'False', 'True'],
    'play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 
             'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
PlayTennis = pd.DataFrame(data)

# Encode categorical features
label_encoders = {}
for column in ['outlook', 'temp', 'humidity', 'windy', 'play']:
    le = LabelEncoder()
    PlayTennis[column] = le.fit_transform(PlayTennis[column])
    label_encoders[column] = le  # Store the encoders for potential inverse transformations

# Separate features and target variable
y = PlayTennis['play']
x = PlayTennis.drop(['play'], axis=1)

# Train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x, y)

# Visualize the Decision Tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                                 feature_names=x.columns,  
                                 class_names=label_encoders['play'].inverse_transform([0, 1]),  # Inverse transform for readable class names
                                 filled=True, rounded=True,  
                                 special_characters=True)  
graph = graphviz.Source(dot_data)

# Display the decision tree
graph.view()  # This will open the decision tree in a PDF viewer, or you can use graph.render() to save it

# Alternatively, if you're in a Jupyter Notebook or Colab, use:
graph


# prac4
# Feed farward backpropagation neural network

import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize random seed for reproducibility
        np.random.seed(42)
        # Randomly initialize weights
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # Activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # Calculate output
            output = self.think(training_inputs)
            # Calculate error
            error = training_outputs - output
            # Adjust weights
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":
    # Initialize the neural network
    neural_network = NeuralNetwork()
    print("Beginning Randomly Generated Weights:")
    print(neural_network.synaptic_weights)

    # Training data: 4 examples with 3 input values and 1 output
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    training_outputs = np.array([[0], [1], [1], [0]])

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training:")
    print(neural_network.synaptic_weights)

    # Get user input
    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering New Situation:", user_input_one, user_input_two, user_input_three)
    print("New Output Data:")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))





# practical 5
# Aim - Support Vector Machine (in Colab) (Diabetes.csv)
# Train SVM model using given dataset and optimize its parameters.
# Evaluate the performance of this SVM model on test data and analyze the result.

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm


#Read CSV File
df = pd.read_csv('https://raw.githubusercontent.com/abhishekky112/aiprac/refs/heads/main/diabetes.csv')
df.head()
df.shape
df.describe()

# Checking for missing values.
df.isnull().values.any()

#Feature Engineering to replace 0 with avg
zero_not_allowed = ["Glucose","BloodPressure","SkinThickness"]
for column in zero_not_allowed:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna = True))
    df[column] = df[column].replace(np.NaN, mean)
df.describe()

# Splitting the dataset into training and testing sets.
x = df.iloc[:, :-2]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)


print("Train size",x_train.shape)
print("Test Size:",x_test.shape)

# Creating the SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
#Test the Model on Test Data
y_pred = clf.predict(x_test)
print("predicted values:-",y_pred)




#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

#Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:",cm)

#Other efficiency metrics
print(classification_report(y_test,y_pred))




# Practical 6: AdaBoost Ensemble Learning
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

# Load the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
seed = 7
num_trees = 30

# Create the AdaBoost model using the SAMME algorithm
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed, algorithm='SAMME')
results = model_selection.cross_val_score(model, X, Y, cv=10)  # Using 10-fold cross-validation
print("AdaBoost Mean Cross-Validation Score:", results.mean())

# Practical 7: Naive Bayes Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
A = iris.data
y = iris.target

# Split the dataset into training and test sets
A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes Classifier
clf = GaussianNB()
clf.fit(A_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(A_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

#Load the football dataset from CSV
dataset = pd.read_csv('https://raw.githubusercontent.com/abhishekky112/aiprac/refs/heads/main/football.csv')

#Seperate features (X) and Labels (y)
A = dataset[['Argentina', 'France']].values
y = dataset['Result'].values

A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=42)

# Create Naive Bayes Classifier
clf = GaussianNB()

# train the classifier on the training data
clf.fit(A_train, y_train)

# Make predictions on test data
y_pred = clf.predict(A_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ",accuracy)

# Pract8: KNN Algorithm with Varying Number of Neighbors
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the Diabetes dataset
url = 'https://raw.githubusercontent.com/abhishekky112/aiprac/refs/heads/main/diabetes.csv'  # Adjust this path if necessary
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Separate features and target variable
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lists to store accuracies
train_accuracies = []
test_accuracies = []
neighbors = range(1, 21)  # Testing for 1 to 20 neighbors

# Loop over different values of n_neighbors
for n in neighbors:
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n)
    
    # Train the classifier
    knn.fit(X_train, y_train)
    
    # Calculate accuracies
    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)
    
    # Store accuracies
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot accuracies vs. number of neighbors
plt.figure(figsize=(12, 6))
plt.plot(neighbors, train_accuracies, marker='o', label='Training Accuracy', color='blue')
plt.plot(neighbors, test_accuracies, marker='o', label='Testing Accuracy', color='orange')
plt.ylim(0, 1)  # Set the y-axis limits
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.grid()  # Optional: Add gridlines for better readability
plt.legend()
plt.show()

# Calculate ROC curve for the best n_neighbors (for example, n=5)
best_n_neighbors = 5  # Adjust this as needed based on your findings
knn_best = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_best.fit(X_train, y_train)
y_scores = knn_best.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


#prac9
# association rule of mining
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Install apyori if not already installed
try:
    import apyori
except ImportError:
    !pip install apyori
from apyori import apriori

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/abhishekky112/aiprac/refs/heads/main/grocery.csv', parse_dates=['Date'], dayfirst=True)

# Display the first few rows
print(df.head())

# Check for missing values
print("Missing values:", df.isnull().any().sum())

# Unique products
all_products = df['itemDescription'].unique()
print("Total products: {}".format(len(all_products)))

# Function to create a distribution plot
def distribution_plot(x, y, name=None, xaxis=None, yaxis=None):
    fig = go.Figure([go.Bar(x=x, y=y)])
    fig.update_layout(
        title_text=name,
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    fig.show()

# Plot top 10 products
top_products = df['itemDescription'].value_counts().nlargest(10)
distribution_plot(x=top_products.index, y=top_products.values, yaxis="Count", xaxis="Products")

# One-hot encoding of items
one_hot = pd.get_dummies(df['itemDescription'])
df.drop('itemDescription', inplace=True, axis=1)
df = df.join(one_hot)

# Create transaction records
records = df.groupby(['Member_number', 'Date'])[one_hot.columns].sum()
records = records.reset_index(drop=True)

# Function to get product names from records
def get_Pnames(x):
    return [product for product in all_products if x[product] > 0]

# Apply the function to get transaction lists
transactions = records.apply(get_Pnames, axis=1).tolist()

# Print the first 10 transactions
print("Sample transactions:", transactions[:10])

# Generate association rules
rules = apriori(transactions, min_support=0.00030, min_confidence=0.05, min_lift=3, min_length=2)
association_results = list(rules)

# Display the results
for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " --> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("===================================")



# practical 10
# tensorflow tools


# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Create sample text files (Spam.txt and NSpam.txt)
spam_content = """Subject: Congratulations! You've Won $100,000 Cash Prize.

Dear George,
I am thrilled to inform you that you are the lucky winner of our recent contest and have been awarded a cash prize of $100,000! Your participation and enthusiasm are truly appreciated, and we couldn't be happier to share this exciting news with you. 

Regards,
idontSmile"""

nspam_content = """Subject: Invitation for Dinner

Dear Friend, 
I hope this email finds you well. I wanted to extend a warm invitation to you for a dinner party on 18/8/2023 at my home next Friday, and it would be wonderful to have you join us.

Best regards, 
iSmile"""

with open("Spam.txt", "w") as f:
    f.write(spam_content)

with open("NSpam.txt", "w") as f:
    f.write(nspam_content)

# Step 2: Prepare a dataset of labeled emails (spam and non-spam)
emails = [
    "Buy cheap watches! Free shipping!",
    "Meeting for lunch today?",
    "Claim your prize! You've won $1,000,000!",
    "Important meeting at 3 PM.",
    "Congratulations! You've won a lottery!",
    "Dinner party invitation.",
    "Earn money fast! Click here!",
    "Your account has been compromised.",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for non-spam

# Step 3: Tokenize and pad the email text data
max_words = 1000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)
X_padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Step 4: Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
training_data = np.array(X_padded)
training_labels = np.array(labels)
model.fit(training_data, training_labels, epochs=10)

# Step 6: Test with 'Spam.txt'
file_path = "./Spam.txt"

with open(file_path, "r", encoding="utf-8") as file:
    sample_email_text = file.read()

sequences_sample = tokenizer.texts_to_sequences([sample_email_text])
sample_email_padded = pad_sequences(sequences_sample, maxlen=max_len, padding="post", truncating="post")
prediction = model.predict(sample_email_padded)
threshold = 0.5

if prediction > threshold:
    print(f"Sample Email ('{file_path}'): SPAM")
else:
    print(f"Sample Email ('{file_path}'): NOT SPAM")

# Step 7: Test with 'NSpam.txt'
file_path = "./NSpam.txt"

with open(file_path, "r", encoding="utf-8") as file:
    sample_email_text = file.read()

sequences_sample = tokenizer.texts_to_sequences([sample_email_text])
sample_email_padded = pad_sequences(sequences_sample, maxlen=max_len, padding="post", truncating="post")
prediction = model.predict(sample_email_padded)

if prediction > threshold:
    print(f"Sample Email ('{file_path}'): SPAM")
else:
    print(f"Sample Email ('{file_path}'): NOT SPAM")
