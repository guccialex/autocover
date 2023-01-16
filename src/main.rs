use std::process::Command;
mod collect;
mod other;
mod predict;
mod getimagesfromstream;
pub use other::*;
use predict::PredictStruct;
use getimagesfromstream::ImageStream;


//create a const called "DIMENSIONS"
pub const DIMENSIONS: (u32, u32) = (64, 64);

pub fn speak(string:&str){
    let string = string.replace(" ", "-");
    Command::new("espeak")
    .arg(string)
    .spawn();
}



pub struct StreamPrediction{
    imagestream: ImageStream,
    predictstruct: PredictStruct,
}

impl StreamPrediction{

    pub fn new(ip: &str) -> StreamPrediction{

        return StreamPrediction{
            imagestream: ImageStream::new(ip),
            predictstruct: PredictStruct::new()
        }
    }

    pub fn poll( &mut self)  -> f32{

        let mut toreturn = 0.0;

        if let Some(image) = self.imagestream.poll(){
            //save the image to a file
            //image.save("test.png").unwrap();
            println!("next image");

            self.predictstruct.submit_next_image(image);
            let prediction = self.predictstruct.commercial_chance();
            println!("prediction: {}", prediction);

            toreturn = prediction;
        }

        return toreturn;
    }

}

use rand::Rng;



pub fn collect_stream( name:&str, ip: &str){

    //create a folder called "streams/name"

    let mut rng = rand::thread_rng();
    let random: String = (0..5).map(|_|  rng.sample(rand::distributions::Alphanumeric)  as char ).collect::<String>() ;

    let path = format!("streams/{}{}", name, random);

    std::fs::create_dir(path.clone()).unwrap();

    let child = save_stream(  &(path+"/"), ip );
}

mod tchclassifier;


//get the numbers that are commercials at what frame rate
//create frames at a much higher frame rate

use std::path::Path;


fn timestamp_screenshots(folder: &Path, secondsperframe: f32){

    for entry in std::fs::read_dir(folder.clone()).unwrap() {

        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {

            let filename = path.file_name().unwrap().to_str().unwrap();

            //println!("filename {}", filename);
            //get the number
            let number = &filename[0..filename.len()-4];
            if number.len() != 6{
                continue;
            }

            //println!("number {}", number);

            let number = number.parse::<f32>().unwrap();

            let seconds = (number * secondsperframe) - secondsperframe * 0.5;
            //println!("seconds {}", seconds);

            let new_name = format!("{:08.2}.jpg", seconds);
            //println!("new name {}", new_name);
            let new_path = path.with_file_name(new_name);
            //println!("path {}, newpath {}", path.to_str().unwrap(), new_path.to_str().unwrap());
            std::fs::rename(path, new_path).unwrap();
        }
    }

}

// //create the "comm" and "noncomm" folders in the directory
// std::fs::create_dir(filepath.parent().unwrap().join("comm"));
// std::fs::create_dir(filepath.parent().unwrap().join("noncomm"));



fn extract_at_framerate(filepath: &Path, extractedpath: &Path, secondsperframe: f32){

    //delete and then create a folder called "extracted" in the folderpath directory
    // let extractedpath = filepath.parent().unwrap().join("extracted");
    // println!("extracting to {}", extractedpath.to_str().unwrap());
    // std::fs::remove_dir_all(extractedpath.clone()).unwrap_or(());
    // std::fs::create_dir(extractedpath.clone());

    //let extractedpath = filepath.parent().unwrap();


    //ffmpeg -i zec.mp4 -vf fps=1/50 -vsync 0 output_%04d.png

    let mut child = Command::new("ffmpeg")
    .arg("-i")
    .arg( filepath.to_str().unwrap() )
    .arg("-vf")
    .arg(format!("fps=1/{}", secondsperframe))
    .arg("-vsync")
    .arg("0")
    .arg( extractedpath.join("%06d.jpg").to_str().unwrap() )
    .spawn()
    .expect("failed to execute process");


    //wait for the child to finish
    let ecode = child.wait().expect("failed to wait on child");
    std::thread::sleep(std::time::Duration::from_secs(5));

    timestamp_screenshots(  &extractedpath, secondsperframe);
}



use std::collections::BTreeMap;
use ordered_float::OrderedFloat;

fn get_time_to_content_type(folderwithcommandnoncomm: &Path) -> BTreeMap< OrderedFloat<f32>, bool>{


    let mut btreemap = BTreeMap::new();


    let commfolder = folderwithcommandnoncomm.join("comm");

    for entry in std::fs::read_dir(commfolder.clone()).unwrap() {

        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {

            let filename = path.file_name().unwrap().to_str().unwrap();
            
            //remove the last 4 digits
            let number = &filename[0..filename.len()-4];
            let number = number.parse::<f32>().unwrap();
            let number = OrderedFloat(number);

            btreemap.insert(number,  true);
        }
    }

    let noncommfolder = folderwithcommandnoncomm.join("noncomm");

    for entry in std::fs::read_dir(noncommfolder.clone()).unwrap() {

        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {

            let filename = path.file_name().unwrap().to_str().unwrap();

            let number = &filename[0..filename.len()-4];
            let number = number.parse::<f32>().unwrap();
            let number = OrderedFloat(number);

            btreemap.insert(number,  false);
        }
    }

    return btreemap;
}



fn rename_folder_by_contents(btreemap: BTreeMap< OrderedFloat<f32>, bool>, imagesfolder: &Path ){


    
    for entry in std::fs::read_dir(imagesfolder  ).unwrap() {

        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {

            let filename = path.file_name().unwrap().to_str().unwrap();

            if filename.contains("comm"){
                continue;
            }

            //println!("filename: {}", filename);
            //get the number
            let number = &filename[0..filename.len()-4];
            let number = number.parse::<f32>().unwrap();
            let number = OrderedFloat(number);

            let previous_value = btreemap.range(..number).next_back().map(|(k, v)| v);

            match previous_value{
                Some(true) => { 

                    let new_name = &filename[0..filename.len()-4];
                    let new_name = format!("{}-comm.jpg", new_name);

                    let new_path = imagesfolder.join(new_name);
                    std::fs::rename(path, new_path).unwrap();
                },
                Some(false) => { 

                    let new_name = &filename[0..filename.len()-4];
                    let new_name = format!("{}-noncomm.jpg", new_name);

                    let new_path = imagesfolder.join(new_name);
                    std::fs::rename(path, new_path).unwrap();
                },
                None => { 
                },
            }


        }
    }




}





fn main() {


    let folderpath = "./mainstreams/zec";
    let filename = "zec5";

    let folderpath = Path::new(folderpath);
    let filepath = folderpath.join(filename).with_extension("mp4");

    //create a new folder with the name of the file
    let extractedpath = folderpath.join(filename);


    let first = false;

    if first{

        std::fs::remove_dir_all(extractedpath.clone()).unwrap_or(());
        std::thread::sleep(std::time::Duration::from_secs(3));
        std::fs::create_dir(extractedpath.clone()).unwrap();
    
        std::fs::create_dir(extractedpath.clone().join("comm") ).unwrap();
        std::fs::create_dir(extractedpath.clone().join("noncomm") ).unwrap();
    
        extract_at_framerate( &filepath, &extractedpath, 5.0 );
    
    }
    else{

        println!("extracted path: {:?}", extractedpath);

        let btreemap = get_time_to_content_type( &extractedpath );
    
        extract_at_framerate( &filepath, &extractedpath, 0.5 );
    
        rename_folder_by_contents(btreemap, &extractedpath);    

    }



    panic!("done");


    //let pathstring = "./mainstreams/zec/zec1.mp4";

    

    // extract_at_framerate( &path, 5.0);



    // create_higher_framerate( path, 0.1 );

    panic!("done");


    panic!("doen");
    //train an image classifier
    //tchclassifier::run();


    //for each image in the folder
    let path = "./mainstreams/zec/zecnoncomm";

    let images = std::fs::read_dir(path).unwrap();
    let images = images.map(|f| f.unwrap().path()).collect::<Vec<_>>();
    let mut images = images.iter().map(|f| f.to_str().unwrap()).collect::<Vec<_>>();


    //order alphabetically
    images.sort();


    let mut count = 0;

    for image in images{


        println!("image {}", image);

        let firstpart = image.split("-").collect::<Vec<_>>()[0];

        count += 1;

        //new name with 5 leading zeroes
        let newname = format!("{}-{:05}.jpg", firstpart, count);
        
        //rename the file
        
        std::fs::rename(image, newname.clone()).unwrap();


    }


    panic!("doine");

    let mut average = (0.0, 0);

    for x in 0..(images.len()-10){


        let image1 = file_to_image( images[x], DIMENSIONS ).unwrap();
        let image2 = file_to_image( images[x+4], DIMENSIONS ).unwrap();

        let difference = other::image_similarity(image1, image2);

        average.0 += difference;
        average.1 += 1;


        if x % 2000 == 1999{
            println!("average: {}", average.0 / average.1 as f32);
            average = (0.0, 0);
        }

    }

    //noncomm
    // average: 0.82526225
    // average: 0.8313134
    // average: 0.84958166


    //comm




    



    panic!("done");



    //turn the video into a series of images

    for x in 2..3{

        let source = format!("./mainstreams/zin/zin{}.mp4", x);
        let output = format!("./mainstreams/zin/images/zin{}-", x);

        println!("saving file {} to {}", source, output);

        save_stream(&source, &output);

        //sleep for 100 seconds
        std::thread::sleep(std::time::Duration::from_secs(20));
    }

    std::thread::sleep(std::time::Duration::from_secs(100));
    
    panic!("done");


    //sort the images into COMM and NONCOMM folder







    let mut child = save_stream(  &("testing/"), "udp://@0.0.0.0:1234" );

    //wait 20 seconds
    std::thread::sleep(std::time::Duration::from_secs(20));

    //kill the child
    child.kill().unwrap();


    panic!("done");


    //get the string contents from "todetect.txt"
    let todetect = std::fs::read_to_string("todetect.txt").unwrap();


    for line in todetect.lines(){
        //split the line into two parts
        let mut parts = line.split(" ");
        //get the first part
        let name = parts.next().unwrap();
        //get the second part
        let path = parts.next().unwrap();


        let fullpath = format!("udp://@239.{}:1234", path);


        collect_stream( name, &fullpath);

        // println!("fullpath {:?}", fullpath);

        // let mut streamprediction = StreamPrediction::new(&fullpath);


        // for x in 0..1500{
        //     let prediction = streamprediction.poll();
        //     println!("prediction {:?}, name{:?}", prediction, name);
        //     if prediction > 0.3{
        //         speak(name);
        //     }
        //     std::thread::sleep(std::time::Duration::from_millis(300));
        // }

    }


    panic!("done");



    //let mut child = other::stream_video("videoplayback.mp4", "0.0.0.0:1234").unwrap();


    // for x in 0..50{

    //     streamprediction.poll();

    //     std::thread::sleep(std::time::Duration::from_millis(1000));
    // }

    // child.kill();

    // panic!("done");
    
}


