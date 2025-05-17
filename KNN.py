import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

file = "heart_data.csv"

def get_data(file):
    data_list = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            data = line.strip().split(',')
            data_list.append(data)
    return data_list

def process_data(data_list):
    unsorted_data = np.array(data_list, dtype=float)
    floored_column = np.floor(unsorted_data[:, 0])
    data = [floored_column, unsorted_data[:, 2], unsorted_data[:, 4], unsorted_data[:, -1]]
    data_array = np.array(data)
    transposed_data = np.transpose(data_array)
    return transposed_data

def get_training_data(transposed_data):
    training_features = np.array([
        transposed_data[:251, 0],  # Age
        transposed_data[:251, 1],  # Creatinine phosphokinase
        transposed_data[:251, 2]   # Ejection fraction
    ])
    training_target = np.array(transposed_data[:251, 3])  # Death event
    return training_features, training_target

def get_test_data(transposed_data):
    test_features = np.array([
        transposed_data[251:, 0],  # Age
        transposed_data[251:, 1],  # Creatinine phosphokinase
        transposed_data[251:, 2]   # Ejection fraction
    ])
    test_target = np.array(transposed_data[251:, 3])  # Actual DEATH_EVENT values for evaluation
    return test_features, test_target

def euclidean_distance(k, test_features, training_features, training_target):
    predictions = []

    # Iterate through each test point
    for x1, y1, z1 in zip(test_features[0], test_features[1], test_features[2]):
        distance_list = []

        # Calculate distance from the current test point to all training points
        for x2, y2, z2 in zip(training_features[0], training_features[1], training_features[2]):
            distance = math.sqrt(((x1 - x2)**2) + ((y1 - y2)**2) + ((z1 - z2)**2))
            distance_list.append(distance)

        # Find the indices of the k smallest distances
        distance_array = np.array(distance_list)
        k_smallest_indices = np.argsort(distance_array)[:k]

        # Get the corresponding DEATH_EVENT values of the k nearest neighbors
        k_nearest_classes = training_target[k_smallest_indices]

        # Predict the most common DEATH_EVENT (modal class)
        most_common = Counter(k_nearest_classes).most_common(1)[0][0]
        predictions.append(most_common)

    return np.array(predictions)

def evaluate_performance(predictions, actual):
    accuracy = np.mean(predictions == actual) * 100
    mse = np.mean((predictions - actual) ** 2)
    return accuracy, mse

def main():
    # Load and process the data
    data_list = get_data(file)
    transposed_data = process_data(data_list)
    training_features, training_target = get_training_data(transposed_data)
    test_features, test_target = get_test_data(transposed_data)

    # Evaluate for different values of k
    for k in [1, 3, 5, 7]:
        predictions = euclidean_distance(k, test_features, training_features, training_target)
        accuracy, mse = evaluate_performance(predictions, test_target)

        print(f"\nFor k = {k}:")
        print(f"Predicted Classes: {predictions}")
        print(f"Actual Classes:    {test_target}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")

main()



