"""Simple PLA coding challenge script"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# Read inputs
def read_inputs():
    """Read user's input according to the format in the example"""
    num_examples, num_features = map(int, input().split())

    # Initialize empty lists to store the feature vectors and class labels
    x = []
    y = []

    # Read the data points line by line
    for _ in range(num_examples):
        values = list(map(float, input().split()))

        # Extract the features
        sample_features = values[:num_features]

        # Extract the class label
        label = int(values[-1])

        # Append the feature vector and class label to the respective lists
        x.append(sample_features)
        y.append(label)

    return np.array(x), np.array(y)


# Normalize feature columns function using min and max values
def min_max_norm(features: np.ndarray) -> np.ndarray:
    """ Perform min-max normalization for each feature column"""
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)
    feature_ranges = max_values - min_values
    normalized_features = (features - min_values) / feature_ranges
    return normalized_features


# Plot the predicted class values from the test set
def plot_results(predictions, y_true, features):
    """Plot results in a 2D plot with feature 1 in x and 
       feature 2 in y axis, class as hue."""
    features_count = features[0].shape[0]

    if features_count == 2:
        feat_idx1 = 0
        feat_idx2 = 1
    else:
        print("Select 2 feature indices to plot results:")
        feat_idx1, feat_idx2 = list(map(int, input().split()))

    correct_x_y = features[predictions == y_true]
    incorrect_x_y = features[predictions != y_true]

    # Split to x,y for plotting
    correct_x = correct_x_y[:, feat_idx1]
    correct_y = correct_x_y[:, feat_idx2]

    incorrect_x = incorrect_x_y[:, feat_idx1]
    incorrect_y = incorrect_x_y[:, feat_idx2]

    # Plot results with feat1 and feat 2 on x and y axis and class as hue
    plt.scatter(correct_x, correct_y, c="green", label="Correctly Classified")
    plt.scatter(incorrect_x, incorrect_y, c="red", label="Incorrectly Classified")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Test set results")
    plt.legend()
    plt.grid(True)

    plt.show()


class Perceptron:
    """A Simple Perceptron"""
    # pylint: disable=R0913
    def __init__(self, x, y, training, trained_weights=None, trained_bias=None):
        """Model initializer
           Input args:
                x : Feature array
                y : True labels
                training: Boolean to indicate training/testing functionality
                trained_weights: Trained weights to load to the model
                trained_bias: Trained bias to load to the model
        """
        # Starting learning rate
        self.learning_rate = 0.001

        # Randomly chosen
        self.max_epochs = 50

        if training is True:
            self.x = x
            self.y = y

            # Initialize the weights to small random values from a normal distribution
            # Bias to 0
            self.weights, self.bias = self.weights_init()
        else:
            self.x = min_max_norm(x)
            self.y = y

            # Load pre-trained weights if provided
            if trained_weights is not None and trained_bias is not None:
                self.weights = trained_weights
                self.bias = trained_bias
            else:
                self.weights, self.bias = self.weights_init()

    # Initialize weights and bias
    def weights_init(self) -> Tuple[np.ndarray, np.ndarray]:
        """Weights init function"""
        self.weights = np.random.randn(self.x.shape[1]) * 0.01
        self.bias = 0
        return (self.weights, self.bias)

    # Train the perceptron model
    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """Training function using the sign activation func"""
        for i in range(self.max_epochs):
            false_predictions = 0
            for x, y in zip(min_max_norm(self.x), self.y):
                # Use the sign activation function :
                # Returns: -1 if pred < 0 | 0 if pred==0 | 1 if x>0
                prediction = np.sign((np.dot(self.weights, x)) + self.bias)

                if prediction != y:
                    self.weights = self.weights + (self.learning_rate * y * x)
                    self.bias = self.bias + (self.learning_rate * y)
                    false_predictions += 1

            # Stop training if the model has managed to linearly separate
            # the two classes (-1,1)
            if false_predictions == 0:
                break
            print(f"Epoch: {i}")
        print("Finished Training...")
        return (self.weights, self.bias)

    # Load pre-trained weights
    def load_weights(self, trained_weights: np.ndarray, trained_bias):
        """Weights Loader"""
        self.weights = trained_weights
        self.bias = trained_bias

    # Predict function
    def predict(self, x: np.ndarray):
        """Use this to predict on a single sample"""
        prediction = np.sign((np.dot(self.weights, x)) + self.bias)
        return prediction

    # Evaluate model on test set
    def evaluate(self) -> float:
        """Model evaluation entry for the test set provided
           as argument to the constructor"""
        correct_predictions = 0
        total_samples = len(self.x)

        for x, y in zip(min_max_norm(self.x), self.y):
            prediction = self.predict(x)
            if prediction == y:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples * 100

        print(f"Correct predicts: {correct_predictions}")
        print(f"Total samples: {total_samples}")
        return accuracy


if __name__ == "__main__":
    print("Expecting train set input:")
    x_train, y_train = read_inputs()

    # Create a model, train it and output the trained weights and bias
    training_model = Perceptron(x=x_train, y=y_train, training=True)

    print("Training...")

    weights, bias = training_model.train()

    print(f"Weights: {weights} | Bias: {bias}")

    print("Trained model...Expecting testing set input:")

    # Read test set from input
    x_test, y_test = read_inputs()

    test_model = Perceptron(
        x=x_test, y=y_test, training=False, trained_weights=weights, trained_bias=bias
    )

    print("Evaluating model on test set...")
    test_accuracy = test_model.evaluate()

    # Report accuracy on test set
    print(f"Accuracy score on testing set: {test_accuracy}")

    # Visualize test_set_predictions
    preds = np.array([test_model.predict(x) for x in min_max_norm(x_test)])

    plot_results(preds, y_test, x_test)
