use anyhow::Result;
use rand::random;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};
use tch::nn::Optimizer;
use tch::vision::dataset::Dataset;

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,

    vs: nn::VarStore,

    combine1: nn::Linear,
    combine2: nn::Linear,
    

    opt: Optimizer,
}

impl Net {
    fn new(vs: nn::VarStore) -> Net {

        let mut opt = nn::Adam::default().build(&vs, 1e-5).unwrap();
        
        //28 x 28 image with 3 channels

        let conv1 = nn::conv2d(&vs.root(), 3, 32, 5, Default::default());
        let conv2 = nn::conv2d(&vs.root(), 32, 64, 5, Default::default());
        let fc1 = nn::linear(&vs.root(), 1024, 1024, Default::default());
        let fc2 = nn::linear(&vs.root(), 1024, 20, Default::default());

        let combine1 = nn::linear(&vs.root(), 40, 20, Default::default());
        let combine2 = nn::linear(&vs.root(), 20, 1, Default::default());

        Net { conv1, conv2, fc1, fc2, combine1, combine2, vs, opt}
    }


    pub fn train(&mut self, firstimage: Vec<f32>, secondimage: Vec<f32>, iscomm: bool) -> f64{

        let firsttensor = Tensor::of_slice(&firstimage).to_kind(tch::Kind::Float);
        let secondtensor = Tensor::of_slice(&secondimage).to_kind(tch::Kind::Float);

        let out = self.full_forward(&firsttensor, &secondtensor, true);
        let out = out.view([1]);

        let prediction = f64::from(out.get(0));



        let iscommtensor = if iscomm { 1 } else { 0 };
        let blabels = Tensor::of_slice(&[iscommtensor]).to_kind(tch::Kind::Float);
        let blabels = blabels.view([1]);

        let loss = out.l1_loss(&blabels, tch::Reduction::Mean);
        
        self.opt.backward_step(&loss);

        return prediction;

    }

    //let imagevector = imagevector.into_iter().flatten().flatten().collect::<Vec<f32>>();

    pub fn predict(&mut self, firstimage: Vec<f32>, secondimage: Vec<f32>) -> f64{
            
        let firsttensor = Tensor::of_slice(&firstimage).to_kind(tch::Kind::Float);
        let secondtensor = Tensor::of_slice(&secondimage).to_kind(tch::Kind::Float);

        let out = self.full_forward(&firsttensor, &secondtensor, false);
        let out = out.view([1]);

        let prediction = f64::from(out.get(0));

        return prediction;
    }



    pub fn full_forward(&self, firstimage: &Tensor, second: &Tensor, train: bool) -> Tensor{

        let first = self.forward_t(firstimage, train);
        let second = self.forward_t(second, train);

        let combined = Tensor::cat(&[first, second], -1);

        combined.apply(&self.combine1).relu().apply(&self.combine2)
    }

}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {

        xs.view([-1, 3, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)

        //what dimensions of image does forward_t expect?

        //it looks like it expects a 4d tensor, and the first dimension is the batch size
        //the second dimension is the number of channels
        //the third dimension is the height
        //the fourth dimension is the width
    }
}

use rand::prelude::SliceRandom;

pub fn get_file_groupings(path: &str) -> Vec<(String, String)>{

    let mut toreturn = Vec::new();

    //"./mainstreams/zet/zetcomm"

    let commfiles = std::fs::read_dir(path).unwrap();
    let commfiles = commfiles.map(|f| f.unwrap().path()).collect::<Vec<_>>();
    let mut commfiles = commfiles.iter().map(|f| f.to_str().unwrap()).collect::<Vec<_>>();

    //order alphabetically
    commfiles.sort();


    loop {
        let file1 = commfiles.remove(0).to_string();
        //x + 1 ahead of file 1
        let file2 = commfiles.remove(5).to_string();
        toreturn.push((file1, file2));
    
        if commfiles.len() < 10 {
            break;
        }
    }


    return toreturn;

}


pub fn run() {

    let device = Device::cuda_if_available();

    println!("device = {:?}", device);
    
    let vs = nn::VarStore::new(device);
    let mut net = Net::new( vs );


    let mut files = Vec::new();

    let binding = get_file_groupings("./mainstreams/zet/zetcomm");
    files.extend( binding.iter().map(|x| (x, true)).collect::<Vec<_>>() );

    let binding = get_file_groupings("./mainstreams/zet/zetnoncomm");
    files.extend( binding.iter().map(|x| (x, false)).collect::<Vec<_>>() );

    let binding = get_file_groupings("./mainstreams/zec/zeccomm");
    files.extend( binding.iter().map(|x| (x, true)).collect::<Vec<_>>() );

    let binding = get_file_groupings("./mainstreams/zec/zecnoncomm");
    files.extend( binding.iter().map(|x| (x, false)).collect::<Vec<_>>() );

    let binding = get_file_groupings("./mainstreams/zin/zincomm");
    files.extend( binding.iter().map(|x| (x, true)).collect::<Vec<_>>() );

    let binding = get_file_groupings("./mainstreams/zin/zinnoncomm");
    files.extend( binding.iter().map(|x| (x, false)).collect::<Vec<_>>() );

    
    files.shuffle(&mut rand::thread_rng());


    let mut average = (0.0, 1);

    for ((firstfile, secondfile), iscomm) in files {

        //used to be 28, 28

        let firstimage = crate::file_to_image(&firstfile, (28,28)).unwrap();
        let firstimage = crate::image_to_tensor_vector(firstimage).into_iter().flatten().flatten().collect::<Vec<f32>>();;

        let secondimage = crate::file_to_image(&secondfile, (28,28)).unwrap();
        let secondimage = crate::image_to_tensor_vector(secondimage).into_iter().flatten().flatten().collect::<Vec<f32>>();;

        let prediction = net.train( firstimage, secondimage, iscomm);
        
        let iscomm = if iscomm { 1 } else { 0 };
        let iscomm = iscomm as f64;

        let accuracy = (iscomm - prediction).abs();
        //let accuracy = if (prediction > 0.5 && iscomm == 1) || (prediction < 0.5 && iscomm == 0) { 1 } else { 0 };
        
        average.0 += accuracy as f64;
        average.1 += 1;





        if average.1 > 500{

            println!("average: {:?}", average.0 / average.1 as f64);
            average.0 = 0.0;
            average.1 = 1;
        
        }

    }




        //     let zincommfiles = std::fs::read_dir("./mainstreams/zin/zincomm").unwrap();
        //     let zincommfiles = zincommfiles.map(|f| f.unwrap().path()).collect::<Vec<_>>();
        //     let zincommfiles = zincommfiles.iter().map(|f| f.to_str().unwrap()).collect::<Vec<_>>();

        //     //filter randomly 90% of the files
        //     let zincommfiles = zincommfiles.iter().filter(|_| { random::<f32>() > 0.9  } ).collect::<Vec<_>>();

        //     let mut averagezincomm = (0.0, 1);

        //     for filename in zincommfiles {

        //         let image = crate::file_to_image(filename, (28,28)).unwrap();
        //         let tensor = crate::image_to_tensor_vector(image);
        //         let prediction = net.predict(tensor);
                
        //         averagezincomm.0 += prediction;
        //         averagezincomm.1 += 1;
        //     }
        //     println!("averagezincomm: {:?}", averagezincomm.0 / averagezincomm.1 as f64);




        //     let zinnoncommfiles = std::fs::read_dir("./mainstreams/zin/zinnoncomm").unwrap();
        //     let zinnoncommfiles = zinnoncommfiles.map(|f| f.unwrap().path()).collect::<Vec<_>>();
        //     let zinnoncommfiles = zinnoncommfiles.iter().map(|f| f.to_str().unwrap()).collect::<Vec<_>>();

        //     //filter randomly 90% of the files
        //     let zinnoncommfiles = zinnoncommfiles.iter().filter(|_| { random::<f32>() > 0.9  } ).collect::<Vec<_>>();

        //     let mut averagezinnoncomm = (0.0, 1);

        //     for filename in zinnoncommfiles {

        //         let image = crate::file_to_image(filename, (28,28)).unwrap();
        //         let tensor = crate::image_to_tensor_vector(image);
        //         let prediction = net.predict(tensor);
                
        //         averagezinnoncomm.0 += prediction;
        //         averagezinnoncomm.1 += 1;
        //     }
        //     println!("averagezinnoncomm: {:?}", averagezinnoncomm.0 / averagezinnoncomm.1 as f64);
        // }



    // }







}