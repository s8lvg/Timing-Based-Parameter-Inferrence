import requests
import json
import random

def send_request_to_network(session,params):
    response = session.post("http://127.0.0.1:8080/predict", json={"input_values" : params})
    return json.loads(response.text)
    
def load_network(session,path):
    response = session.post("http://127.0.0.1:8080/loadmodel", json={"path": path})
    return response.text

def random_parameters_request(session):
    params = [random.uniform(0,1) for _ in range(4)]
    return_dict = send_request_to_network(session,params)
    return_dict["input_values"] = params
    return return_dict


def main():
    # Start a session 
    session = requests.session()
    # Request random inputs and print timing
    print(f"Prediction Time: {random_parameters_request(session)['prediction_time']}")

if __name__=="__main__":
    main()