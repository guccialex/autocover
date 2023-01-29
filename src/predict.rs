
use tract_onnx::prelude::*;
use image::DynamicImage;
use crate::image_to_tensor_vector;
use crate::file_to_image;
use ndarray::Axis;

pub struct ResNet{
    resnetmodel: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    commmodel: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,

}


impl ResNet{

    pub fn new() -> ResNet{

        let resnetmodel = tract_onnx::onnx()
        // load the model
        .model_for_path("models/resnet.onnx").unwrap()
        // optimize the model
        .into_optimized().unwrap()
        // make the model runnable and fix its inputs and outputs
        .into_runnable().unwrap();

        let commmodel = tract_onnx::onnx()
        .model_for_path("models/cnn.onnx").unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();

        return ResNet{
            resnetmodel: resnetmodel,
            commmodel: commmodel,
        };
    }

    pub fn image_through_resnet(&self, imagepath: &str) -> ndarray::Array1<f32>{

        let image = file_to_image(imagepath, (224, 224));
        let tensor = image_to_tensor_vector( image.unwrap() );
        let tensor = tensor.into_iter().flatten().flatten().collect::<Vec<f32>>();

        let tensor = ndarray::Array::from_shape_vec((1, 3, 224, 224), tensor).unwrap();
        let tensor: Tensor = tensor.into();
    
        let result = self.resnetmodel.run(tvec!(tensor) ).unwrap();

        //convert to an ndarray::Array
        let tensor = result[0].clone().into_tensor();

        //convert to an ndarray::Array1
        let tensor = tensor.into_array::<f32>().unwrap();

        //flatten the array
        let tensor = tensor.into_shape(1000).unwrap();

        return tensor;
    }

    pub fn predict(&self, imagepath1: &str, imagepath2: &str) -> f64{

        let tensor1 = self.image_through_resnet(imagepath1);
        let tensor2 = self.image_through_resnet(imagepath2);

        //stack the tensors
        let tensor = ndarray::stack(Axis(0), &[tensor1.view(), tensor2.view()]).unwrap();
        //unsqueeze
        let tensor = tensor.insert_axis(Axis(0));
        let tensor: Tensor = tensor.into();

        //feed into comm model
        let result = self.commmodel.run( tvec!(tensor) ).unwrap();
        
        // //get the shape of the result
        // let shape = result[0].shape();
        // println!("shape: {:?}", shape);

        let x: f64 = result[0].clone().cast_to_scalar().unwrap();
        
        return x;

        // //let image = file_to_image("test.png", (64, 64));
        // let tensor = image_to_tensor_vector( image );
        // let tensor = tensor.into_iter().flatten().flatten().collect::<Vec<f32>>();
        // let tensor = ndarray::Array::from_shape_vec((1, 3, 64, 64), tensor).unwrap();
        // let tensor: Tensor = tensor.into();
    
        // let result = self.model.run(tvec!(tensor) ).unwrap();
        // let x: f64 = result[0].clone().cast_to_scalar().unwrap();
        // x
    }

}



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