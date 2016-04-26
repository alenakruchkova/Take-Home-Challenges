import json
import random
import numpy as np
from sklearn import linear_model


############### DATA PREPROCESSING #####################


def open_file(csv_file):
    """Opens csv file and adds information for each
    business to train_data list"""

    train_data = []

    for data in open(csv_file):

        row = data.splitlines()     # row is a list with a string of all data for a company

        for string in row:
            unique_id, city, state, contact_title, category, PRMKTS, EQMKTS, RAMKTS, MMKTS, has_facebook, has_twitter, degree_connected, revenue, headcount = string.split(",")

            # businesses missing degree_connected data are assigned value of 0
            degree_connected = degree_connected or 0

            train_data.append([unique_id, city, state, contact_title, category, PRMKTS, EQMKTS,
            RAMKTS, MMKTS, has_facebook, has_twitter, degree_connected,
            revenue, headcount])

    return train_data


def import_labels(labels):
    """Opens json with labels and returns it as a dict"""

    with open(labels) as data_file:
        json_dict = json.load(data_file)

    return json_dict


cities = {}
states = {}
contacts = {}
categories = {}


def categorical_data_conversion(train_data):
    """ Converts all categorical data into numeric data"""

    for i in range(1, len(train_data)):

        # convert cities to numeric values
        city = train_data[i][1]
        if city in cities:
            pass
        else:
            cities[city] = len(cities)

        # convert states to numeric values
        state = train_data[i][2]
        if state in states:
            pass
        else:
            states[state] = len(states)

        # convert contact titles to numeric values
        contact = train_data[i][3]
        if contact in contacts:
            pass
        else:
            contacts[contact] = len(contacts)

        # convert contact categories to numeric values
        category = train_data[i][4]
        if category in categories:
            pass
        else:
            categories[category] = len(categories)


def convert_into_vector(train_data, json_dict):
    """Takes in list of numeric data and returns a list of data vectors"""

    model_ready_data = []

    for i in range(1, len(train_data)):
        data_vector = []

        # unique_id
        unique_id = train_data[i][0]
        if not unique_id in json_dict:
            # if a row in train dataset is missing a label a random label is assigned
            data_vector.append(random.randint(0, 1))
        else:
            data_vector.append(json_dict[unique_id])

        # city
        city = train_data[i][1]
        city_idx = cities[city]
        one_hot_vector = convert_to_one_hot(city_idx, 20)
        data_vector.extend(one_hot_vector)

        # state
        state = train_data[i][2]
        state_idx = states[state]
        one_hot_vector = convert_to_one_hot(state_idx, 20)
        data_vector.extend(one_hot_vector)

        # contact
        contact = train_data[i][3]
        contact_idx = contacts[contact]
        one_hot_vector = convert_to_one_hot(contact_idx, 10)
        data_vector.extend(one_hot_vector)

        # category
        category = train_data[i][4]
        category_idx = categories[category]
        one_hot_vector = convert_to_one_hot(category_idx, 10)
        data_vector.extend(one_hot_vector)

        # PRMKTS
        prmkts = train_data[i][5]
        data_vector.append(float(prmkts))

        # EQMKTS
        eqmkts = train_data[i][6]
        data_vector.append(float(eqmkts))

        # RAMKTS
        ramkts = train_data[i][7]
        data_vector.append(float(ramkts))

        # MMKTS
        mmkts = train_data[i][8]
        data_vector.append(float(mmkts))

        # facebook
        facebook = train_data[i][9]
        facebook_idx = convert_bool_values(facebook)
        data_vector.append(facebook_idx)

        # twitter
        twitter = train_data[i][10]
        twitter_idx = convert_bool_values(twitter)
        data_vector.append(twitter_idx)

        # degree_connected
        degree_connected = train_data[i][11]
        data_vector.append(float(degree_connected))

        # revenue
        revenue = train_data[i][12]
        data_vector.append(int(revenue))

        # headcount
        headcount = train_data[i][13]
        data_vector.append(int(headcount))

        model_ready_data.append(data_vector)

    return model_ready_data


# HELPER FUNCTIONS FOR CONVERTING DATA


def convert_to_one_hot(i, max_categories):
    """Converts n to one hot vector"""

    enc = [0]*max_categories
    enc[i] = 1
    return enc


def convert_bool_values(n):
    """Converts boolean value to numeric"""

    if n:
        return 1
    else:
        return 0


########### FITING A MODEL / CROSS-VALIDATION ###########

def evaluate_model(data):
    """Fits a model on a subset of the training set and evaluates on the rest."""

    data = np.array(data, dtype=np.float32)

    # Using 2500 first rows as a train set
    train_labels = data[:2500, 0:1]
    train_features = data[:2500, 1:]

    # Using 500 last rows as a test set
    test_labels = data[2500:, 0:1]
    test_features = data[2500:, 1:]

    regr = linear_model.LinearRegression()
    regr.fit(train_features, train_labels)

    # Print out to see what parramenters can be disregarded
    (regr.coef_)

    # Calculate the mean square error
    np.mean((regr.predict(test_features) - test_labels)**2)

    # Evaluate estimator performance
    answers = regr.predict(test_features) > 0.5
    total = 0
    correct = 0
    for myanswer, trueanswer in zip(answers, test_labels):
        if int(myanswer) == int(trueanswer):
            correct += 1
        total += 1

    percentage = (correct * 100)/total

    print "PREDICTION ACCURACY:", "Correct answers:", correct, "Total answer:", total, "Accuracy:", percentage, "%"

############# CALCULATE PREDUCTIONS #######################


def predictions(train_data, test_data):
    """Fit linear model to train_data, return predictions on test_data"""

    train_data = np.array(train_data, dtype=np.float32)

    train_labels = train_data[:, 0:1]
    train_features = train_data[:, 1:]

    test_data = np.array(test_data, dtype=np.float32)
    test_features = test_data[:, 1:]

    regr = linear_model.LinearRegression()
    regr.fit(train_features, train_labels)

    answers = regr.predict(test_features) > 0.5

    return answers


def predictions_json(predictions, test_data, file):
    """Calculates number of "True" and "False" predictions.

    And prints them out (sanity check).
    Creates dictionary with unique_id and correseponding prediction as key-value pairs.
    Writes the dictionary directly to json file."""

    t = 0
    f = 0
    for i in range(len(predictions)):
        if int(predictions[i]) == 1:
            t += 1
        else:
            f += 1

    print "PREDICTIONS:", "True:", t, "False:", f

    answers = {}
    for i in range(1, len(test_data)):
        answers[test_data[i][0]] = int(predictions[i - 1])

    json.dump(answers, open(file, 'wb'))

                   #### DATA PREP AND MODEL FITTING FUNCTION CALLS ###

# open training data
input_data = open_file("data/DS_train.csv")
# import labels for train data
json_dict = import_labels("data/DS_train_labels.json")
# convert categorical data (store conversions in dicts)
# category->numeric mapping is stored in global variables
# in case they are needed later for more conversion
categorical_data_conversion(input_data)
# convert all data into vectors
model_data = convert_into_vector(input_data, json_dict)
# fit the model and print some statistics
evaluate_model(model_data)

                    ### USE FITTED MODEL FOR PREDICTIONS FUNCTION CALLS ###

# open test data
prediction_data = open_file("data/DS_test.csv")
# run in case test data contains information
# not present in training dataset (e.g. new city)
categorical_data_conversion(prediction_data)
# convert all prediction data inro vecors, will not modify unique_id
test_data = convert_into_vector(prediction_data, json_dict)
# using fitted model predict test_data outcomes
predictions = predictions(model_data, test_data)
# generate a json file with test predictions
predictions_json(predictions, prediction_data, "AKruchkova_predictions.json")
