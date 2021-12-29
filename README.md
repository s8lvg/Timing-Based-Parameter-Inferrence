# Mlcysec End Of Term Project

## Iris Net
A simple implementation of a neural network that does classification on the iris dataset. The resulting network gets dumped in onnx format. 

## Timing Blackbox
A rust network evaluator that takes the onnx implemtation of iris_net (or with small modifications any other neural network). Run it with ```cargo +nightly run``` while in the ```timing_blackbox``` directory. It consists of the following parts. 

### Webserver
Handles incoming request to the neural network all communication is done in json format and works like this
- Client sends a post request of the form 
```json
{"input_values" : [<float>,<float>,<float>,<float>]}
``` 
to the prediction API at ```http://127.0.0.1/predict```
- Sever processes request and sends a response in the form 
```json
{"confidence_vector":[<float>,<float>,<float>],"prediction_time":<float>}
```
back to the client

### Prediction API
The prediction api executed the loaded network and returns a confidence vector. In addition execution time of the network is measured in ns 
```rust
// Start timestamp
let start = rdtscp();
// Model gets executed
let result = model_extract.run(tvec!(input_tensor)).unwrap();
// End timestamp
let end = rdtscp();
```
This prediction time then gets send to the client.

## Advesary
A sample implementation that tries to maximise the time it takes for the network to execute. It works by using a genetic algorithm in the following way
- Sends request to the Prediction api
- Uses prediction_time value to compute a fitness function
- Mutate seeds to optimize fitness using genetic algorithm

# Troubleshoot
The timing blackbox uses the assembly instruction ```rdscp``` which may not be availiable on all systems. In case this leads to an error replace all calls to ```rdscp()``` with calls to ```rdsc()```.

In order to debug the timing blackbox start it with the following arguments
```
RUST_LOG=trace cargo +nightly run
```

You can manually query the prediction server from the command line for debug purposes using 
```
wget -O- --post-data='{"input_values":[0.0, 0.0, 0.0, 0.0]}'   --header='Content-Type:application/json'  'http://localhost:8080/predict' > result
```
Get the results using
```
cat result
```