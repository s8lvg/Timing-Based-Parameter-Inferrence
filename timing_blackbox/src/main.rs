extern crate tract_onnx;
extern crate image;
extern crate actix_web;
extern crate serde;
extern crate serde_json;
extern crate structopt;
extern crate exitfailure;
extern crate anyhow;
#[macro_use] extern crate log;
use actix_web::{web, middleware::Logger, App, HttpServer, Result, HttpResponse};
use serde::{Deserialize, Serialize};
use tract_onnx::prelude::*;
use std::arch::asm;
use tract_ndarray::Array;
use std::sync::*;

type ModelType = tract_onnx::prelude::SimplePlan<tract_onnx::prelude::TypedFact, std::boxed::Box<dyn tract_onnx::prelude::TypedOp>, tract_onnx::prelude::Graph<tract_onnx::prelude::TypedFact, std::boxed::Box<dyn tract_onnx::prelude::TypedOp>>>;

// Output of networ in json
#[derive(Serialize)]
struct Prediction {
    confidence_vector: Vec<f32>,
    prediction_time: u64,
}

// Json input to network
#[derive(Deserialize)]
struct Input {
    input_values: Vec<f32>,
}

// Implementation of a precise timestamp using the RDTSCP instruction
#[inline]
fn rdtscp() -> u64{
    let a : u64;
    let d : u64;
    let res : u64;
    unsafe{
        asm!("nop");
        asm!("mfence");
        asm!("rdtscp",
             out("eax") a,
             out("edx") d,
             lateout("rcx") _ ,
            );
        res = (d << 32) | a;
        asm!("mfence");
    }
    return res;
}

// Implementation of a precise timestamp using the RDTSC instruction use this if 
// RDSCP does not work
#[inline]
fn rdtsc() -> u64{
    let a : u64;
    let d : u64;
    let res : u64;
    unsafe{
        asm!("nop");
        asm!("mfence");
        asm!("rdtsc",
             out("eax") a,
             out("edx") d,
            );
        res = (d << 32) | a;
        asm!("mfence");
    }
    return res;
}

async fn make_prediction(model: web::Data<Arc<Mutex<ModelType>>>,input: web::Json<Input>) -> HttpResponse{
    let model_extract = model.lock().unwrap();
    let input_tensor : Tensor = Array::from_shape_vec((1,4),input.input_values.clone()).unwrap().into();

    // Get timed run of model 
    let start = rdtscp();
    let result = model_extract.run(tvec!(input_tensor)).unwrap();
    let end = rdtscp();
    
    // Get timing difference
    let delta = end-start;
    
    // Get result json
    let mut result_probs : Vec<f32> = vec![0.0,0.0,0.0];
    for (i, score) in result[0]
        .to_array_view::<f32>()
        .unwrap()
        .iter()
        .take(5)
        .enumerate() {
            result_probs[i] = *score;
    }

    let prediction = Prediction{
        confidence_vector : result_probs.to_owned(),
        prediction_time : delta.to_owned(),
    };

    HttpResponse::Ok().json(prediction)
}

#[actix_web::main]
async fn main() ->anyhow::Result<()>{

    // Path to model
    let path = "iris_net.onnx";
    // Input shape of model
    let shape = tvec!(1,4);

    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), shape))?
        .into_optimized()?
        .into_runnable()?;
        
    //Create a timestamped logger
    pretty_env_logger::init_timed();
    info!("[+] Loaded model");
    info!("[+] Starting Server at 127.0.0.1:8080");

    HttpServer::new(move || {
        let data = Arc::new(Mutex::new(model.clone()));
        App::new()
            .data(data)
            .service(
                web::resource("/predict").route(
                    web::post().to(make_prediction)))
    })
    .bind("127.0.0.1:8080")
    .unwrap() 
    .run()
    .await;
    Ok(())
}