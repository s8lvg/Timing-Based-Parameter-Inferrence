import requests
import json
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

MAX_NUM_LAYERS = 10
NUM_MEASUREMENTS = 10000
PROFILE = False
NUM_TESTS = 500
TEST_REPEAT = 200


def avg(list):
    return sum(list) / len(list)


def flatten(t):
    return [item for sublist in t for item in sublist]


def send_request_to_network(session, params):
    response = session.post(
        "http://127.0.0.1:8080/predict", json={"input_values": params}
    )
    return json.loads(response.text)


def load_network(session, path):
    response = session.post("http://127.0.0.1:8080/loadmodel", json={"path": path})
    return response.text


def random_parameters_request(session):
    params = [random.uniform(0, 1) for _ in range(4)]
    return_dict = send_request_to_network(session, params)
    return_dict["input_values"] = params
    return return_dict


# Profiles the neural net by performing repeated measurements with different depth.
def profile(session):
    layers_to_timing = [[] for _ in range(MAX_NUM_LAYERS + 1)]

    for layer_cnt in range(1, MAX_NUM_LAYERS + 1):
        # Load new network with layer_cnt layers
        network_loaded = load_network(
            session, f"models/iris_{layer_cnt}_relu_cross-entropy.onnx"
        )
        print(f"[+] Loaded Netowork: {network_loaded}")
        # Measure NUM_MEASUREMENTS times with random inputs
        for _ in range(NUM_MEASUREMENTS):
            layers_to_timing[layer_cnt].append(
                random_parameters_request(session)["prediction_time"]
            )
        # Compute average of measurements
        print(f"    Average timing: {avg(layers_to_timing[layer_cnt])}")

    print("[+] Dumping profiling measurements")
    pickle.dump(layers_to_timing, open("layers_timing_dict.pl", "wb"))
    return layers_to_timing


def generate_testset(session):
    # Generate a on the fly testset by quering the network
    test_y = []
    test_x = []
    for _ in range(NUM_TESTS):
        layer_cnt = random.randint(1, MAX_NUM_LAYERS)
        network_loaded = load_network(
            session, f"models/iris_{layer_cnt}_relu_cross-entropy.onnx"
        )
        print(f"[+] Loaded Netowork: {network_loaded}")

        test_y.append(layer_cnt)
        avg = 0
        for _ in range(TEST_REPEAT):
            avg += random_parameters_request(session)["prediction_time"]
        test_x.append(avg / TEST_REPEAT)

    return np.array(test_x), np.array(test_y)


def main():
    # Start a session
    session = requests.session()

    # The goal of this advasary is to map from the models timing to the number of layers it has

    # To achieve this goal it first queries models with a known layer count and collects an average execution
    # times. Afterwards it a linear regressor is trained on the results as we strongly assume that layer count
    # and execution time are linearly correlated.
    if PROFILE:
        layers_to_timing = profile(session)
    else:
        layers_to_timing = pickle.load(open("layers_timing_dict.pl", "rb"))

    print("[+] Loading Data")
    x = np.array(flatten(layers_to_timing)).reshape(-1, 1)
    y = np.array(
        flatten([[i] * NUM_MEASUREMENTS for i in range(1, MAX_NUM_LAYERS + 1)])
    )

    print("[+] Training LogisticRegression")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=42, multi_class="ovr", n_jobs=16, verbose=1, solver="saga"
        ),
    )
    clf.fit(x, y)

    print("[+] Generating test set")
    test_X, test_y = generate_testset(session)
    pred_y = clf.predict(test_X.reshape(-1, 1))

    print(test_y)
    print(pred_y)
    score = clf.score(test_X.reshape(-1, 1), test_y)
    print(f"[!] Score: {score} ")

    cm = metrics.confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=0.5, square=True, cmap="Blues_r")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    all_sample_title = "Accuracy Score: {0}".format(score)
    plt.title(all_sample_title, size=15)
    plt.show()


if __name__ == "__main__":
    main()
