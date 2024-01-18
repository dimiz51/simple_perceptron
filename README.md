# Simple Perceptron Learning Algorithm (PLA) Coding Challenge

This is a Python script that implements the Perceptron Learning Algorithm (PLA) for a binary classification problem. The script allows you to train a simple perceptron model on a training dataset and evaluate its performance on a testing dataset. Additionally, it provides a visualization of the classification results in a 2D plot.

## How to Run the Script

1. Ensure you have Python installed on your system.

2. Clone or download this repository to your local machine.

3. Open a terminal or command prompt and navigate to the directory where the script is located.

4. Run the script using the following command:
    `python simple_pla.py`

5. Follow the on-screen prompts to provide input for the training and testing datasets.

## Input Format

The script expects input in the following format:

```
<num_examples> <num_features>
<feature1_value> <featurex_value> <class_label>
<feature1_value> <featurex_value> <class_label>
```

Where: \
    **num_examples**: The number of input samples in the dataset. \
    **num_features**: The number of features for each input sample. \
    __feature1_value, feature2_value, ...__: The values of the features for each input sample. \
    **class_label**: The class label for each input sample, which should be either -1 or 1 for binary classification. 

## Example input
Here is an example training set:
```
6 2
77 66 1
86 14 1
50 21 -1
42 82 1
28 78 1
22 32 -1
```
And a testing set:
```
10 2
23 47 -1
26 2 -1
96 78 1
1 65 -1
64 40 1
13 68 1
24 5 -1
69 32 1
21 19 -1
65 88 1
```

## Output
- The script will display the accuracy of the perceptron model on the testing dataset.
- It will also generate a 2D plot to visualize the classification results. The plot shows correctly classified points in green and incorrectly classified points in red.
- If you input more than 2 features you will be asked to provide two feature indices for the plotting.

