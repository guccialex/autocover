
use tract_onnx::prelude::*;
use image::DynamicImage;
use crate::image_to_tensor_vector;

pub struct CommModel{
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl CommModel{

    pub fn new() -> CommModel{

        let model = tract_onnx::onnx()
        // load the model
        .model_for_path("cnn.onnx").unwrap()
        // optimize the model
        .into_optimized().unwrap()
        // make the model runnable and fix its inputs and outputs
        .into_runnable().unwrap();

        return CommModel{
            model: model,
        };
    }

    pub fn predict(&self, image: DynamicImage) -> f64{

        //let image = file_to_image("test.png", (64, 64));
        let tensor = image_to_tensor_vector( image );
        let tensor = tensor.into_iter().flatten().flatten().collect::<Vec<f32>>();
        let tensor = ndarray::Array::from_shape_vec((1, 3, 64, 64), tensor).unwrap();
        let tensor: Tensor = tensor.into();
    
        let result = self.model.run(tvec!(tensor) ).unwrap();
        let x: f64 = result[0].clone().cast_to_scalar().unwrap();
        x
    }

}


pub struct PredictStruct{
    commmodel: CommModel,
    lastpredictions: Vec<f32>,
}

impl PredictStruct{

    pub fn new() -> PredictStruct{
        return PredictStruct{
            commmodel: CommModel::new(),
            lastpredictions: Vec::new(),
        };
    }

    pub fn submit_next_image(&mut self, image: DynamicImage){

        let prediction = self.commmodel.predict(image);
        self.lastpredictions.push(prediction as f32);
    }

    pub fn commercial_chance(&self) -> f32{
        let window = 10;
        //get the average of the last predictions in the window
        let lastpredicitonsinwindow = self.lastpredictions.iter().rev().take(window).map(|x|*x).collect::<Vec<f32>>();
        return lastpredicitonsinwindow.iter().sum::<f32>() / self.lastpredictions.len() as f32;
    }

}