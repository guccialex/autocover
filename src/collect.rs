
use std::path::PathBuf;
use std::sync::mpsc::channel;
use rand::Rng;

use image::{DynamicImage, GrayImage, RgbImage, GenericImageView};

use image::Pixels;
use image::Rgb;
use std::collections::HashMap;


use crate::save_stream;
use crate::file_to_image;

pub fn collect_streams(){

    let (tx, rx) = channel();
    
    ctrlc::set_handler(move || tx.send(()).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");    

    //read the contents of channeltoip.txt
    let contents = std::fs::read_to_string("channeltoip.txt")
        .expect("Something went wrong reading the file");

    let mut children = Vec::new();

    //let random number
    let rand = rand::random::<u16>();

    std::fs::create_dir( format!("./collected{}", rand) ).unwrap();

    for line in contents.split("\n"){

        let x: Vec<&str> = line.split("\t").collect();

        std::fs::create_dir( format!("./collected{}/{}", rand, x[0]) ).unwrap();
        std::fs::create_dir( format!("./collected{}/{}/rx", rand, x[0]) ).unwrap();
        std::fs::create_dir( format!("./collected{}/{}/pgm", rand, x[0]) ).unwrap();

        println!("{:?}",x);
        children.push( save_stream(&format!("./collected{}/{}/rx/", rand, x[0]), x[1]) );
        children.push( save_stream(&format!("./collected{}/{}/pgm/", rand, x[0]), x[2]) );
    }

    rx.recv().expect("Could not receive from channel.");

    println!("cmd exiting ..."); 

    for child in children.iter_mut(){
        child.kill().unwrap();
    }
    
    println!("exiting ..."); 
    
    std::thread::sleep(std::time::Duration::from_secs(2));
}


fn image_differences( image1: &DynamicImage, image2: &DynamicImage) -> f32{

    //get the dimensions
    if image1.dimensions() != image2.dimensions(){
        panic!("Images are not the same size");
    };

    let image1 = image1.to_rgb16();
    let image1 = image1.pixels();
    let image1 = image1.map(|pixel|  vec![pixel[0], pixel[1], pixel[2]]   ).flatten().map(|v| v as f32 / 65536.0).collect::<Vec<_>>();

    let image2 = image2.to_rgb16();
    let image2 = image2.pixels();
    let image2 = image2.map(|pixel|  vec![pixel[0], pixel[1], pixel[2]]   ).flatten().map(|v| v as f32 / 65536.0).collect::<Vec<_>>();

    let sum = image1.iter().zip(image2.iter()).map(|(a,b)| (a-b).abs() ).sum::<f32>() / (image1.len() as f32);

    return sum;
}

pub struct GetComms{
    //the list of previous images
    images: Vec<DynamicImage>,
    //the list of previous similarities
    similarities: Vec<f32>,
}

impl GetComms{

    fn new() -> GetComms{
        GetComms{
            images: Vec::new(),
            similarities: Vec::new(),
        }
    }

    fn add( &mut self, rx: DynamicImage, similarity: f32 ){

        self.similarities.push(similarity);

        self.images.push( rx );

        self.save_comms_and_non_comms();
    }


    fn save_comms_and_non_comms(&mut self) {

        if self.similarities.len() < 601{
            return ();
        }


        while self.images.len() > 600{
            self.images.remove(0);
        }

        while self.similarities.len() > 600{
            self.similarities.remove(0);
        }


        let average_300 = self.similarities.iter().take(600).sum::<f32>() / 600.0;

        let average_10 = self.similarities.iter().take(20).sum::<f32>() / 20.0;



        if rand::thread_rng().gen_range(1..100) > 97{
            println!("average_300: {:?}, average_10: {:?}", average_300, average_10);
        }


        if average_300 < 0.18 && average_10 > 0.18{
            let randomnumber = rand::thread_rng().gen_range(1..100000000).to_string();
            self.images[0].save(format!("comms/{}.png", randomnumber) ).unwrap();
        }
        else{
            let randomnumber = rand::thread_rng().gen_range(1..100000000).to_string();
            self.images[0].save(format!("noncomms/{}.png", randomnumber) ).unwrap();
        }


    }

}



fn get_comm_and_non_comm(){

    //delete the directory called "comms"
    std::fs::remove_dir_all("comms").unwrap_or(());
    std::fs::remove_dir_all("noncomms").unwrap_or(());
    
    // //create a folder here called "comms"
    let commpath = PathBuf::from("./comms");
    std::fs::create_dir(commpath.clone()).unwrap();

    let noncommpath = PathBuf::from("./noncomms");
    std::fs::create_dir(noncommpath.clone()).unwrap();
    

    for folder in std::fs::read_dir("./collectedfolder").unwrap(){


        let folder = folder.unwrap();
        let folder_path = folder.path();
        let folder_name = folder_path.file_name().unwrap().to_str().unwrap();
        println!("folder: {:?}", folder_name);

        if folder_name == "TVP"{
            continue;
        }

        //get every file in this folder
        let dir = std::fs::read_dir(folder_path.clone()).unwrap();
        
        //the number to the path
        let mut rxfiles = HashMap::new();
        let mut pgmfiles = HashMap::new();

        let rxfolder = PathBuf::from(folder_path.clone().to_str().unwrap().to_owned() + "/rx");
        let rxdir = std::fs::read_dir(rxfolder.clone()).unwrap();

        for file in rxdir{
            let file = file.unwrap();
            let file_path = file.path();
            let file_name = file_path.file_name().unwrap().to_str().unwrap();
            let file_name = file_name.split('.').next().unwrap();
            let file_name = file_name.parse::<i32>().unwrap();

            rxfiles.insert(file_name, file_path);
        }


        let pgmfolder = PathBuf::from(folder_path.clone().to_str().unwrap().to_owned() + "/pgm");
        let pgmdir = std::fs::read_dir(pgmfolder.clone()).unwrap();

        for file in pgmdir{
            let file = file.unwrap();
            let file_path = file.path();
            let file_name = file_path.file_name().unwrap().to_str().unwrap();
            let file_name = file_name.split('.').next().unwrap();
            let file_name = file_name.parse::<i32>().unwrap();

            pgmfiles.insert(file_name, file_path);
        }


        let mut rxfilesinorder = rxfiles.clone().into_iter().collect::<Vec<(i32, PathBuf)>>();
        rxfilesinorder.sort_by(|a, b| a.0.cmp(&b.0));


        let mut getcomms = GetComms::new();

        


        let mut lastlargerx = Vec::new();
        let mut lastrx = Vec::new();
        let mut lastpgm = Vec::new();

        let mut idealgap = None;

        for (filenumber, path1) in rxfilesinorder.clone(){

            let matchingnumber = filenumber ;

            if let Some(path2) = pgmfiles.get(&matchingnumber){

                let rx = file_to_image(path1.to_str().unwrap(), (20, 15)).unwrap();
                let pgm = file_to_image(path2.to_str().unwrap(), (20, 15)).unwrap();


                lastrx.push(rx.clone());
                lastpgm.push(pgm.clone());

                while lastrx.len() > 30{
                    lastrx.remove(0);
                }

                while lastpgm.len() > 30{
                    lastpgm.remove(0);
                }
                

                let largerx = file_to_image(path1.to_str().unwrap(), (64, 64)).unwrap();
                lastlargerx.push(largerx.clone());
                while lastlargerx.len() > 30{
                    lastlargerx.remove(0);

                }


                if let Some(curgap) = get_ideal_gap( &lastrx, &lastpgm){
                    idealgap = Some(curgap);
                }


                if let Some(idealgap) = idealgap{

                    let difference = image_differences(&lastrx[0], &lastpgm[idealgap]);

                    getcomms.add(lastlargerx[0].clone(), difference);    
                }


            }
        }



    }
}



//get the best image skip length


//given the last 30 seconds of rx and pgm images, get the estimated seconds between them
//the similarity score
//and the rx image for that
fn get_ideal_gap( rx: &Vec<DynamicImage>, pgm: &Vec<DynamicImage> ) -> Option<(usize)>{

    let skiprange = 10;

    if rx.len() < 30{ return None; }
    if pgm.len() < 30{ return None; }

    let mut best_similarity = (0, 10000.0);

    //test skip length of 3 to 10
    for skiplength in 0..skiprange{

        let mut similaritiessum = 0.0;

        for x in 0..rx.len()-skiprange{

            let rx = rx[ x ].clone();
            let pgm = pgm[ x + skiplength ].clone();

            let similarity = image_differences(&rx, &pgm);

            similaritiessum += similarity;
        }

        similaritiessum = similaritiessum / (rx.len()-skiprange) as f32;

        best_similarity = if similaritiessum < best_similarity.1{
            (skiplength, similaritiessum)
        }else{
            best_similarity
        };
    }

    //println!("best similarity: {:?}", best_similarity);

    if best_similarity.1 > 0.13{
        return None;
    }
    else{
        return Some( best_similarity.0 );
    }


    // let mut firstdifference = 0.0;
    // //the similarity between the newest pgm and the
    // for x in 0..5{
    //     firstdifference += image_differences(&rx[ x ], &pgm[ x + best_similarity.0 ]);   
    // }
    // firstdifference = firstdifference / 5.0;



    //the image to return should be in the middle
    //return Some( (best_similarity.0 as u32, firstdifference) );
}
