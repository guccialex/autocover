use std::collections::HashMap;
use std::collections::HashSet;
use std::process::Child;
use std::process::Command;

use std::io::Error;

pub fn stream_video(path: &str, ip: &str) -> Result<Child,Error> {

    let mut child = Command::new("ffmpeg")
    .arg("-re")
    .arg("-i")
    .arg(path)
    .arg("-v")
    .arg("0")
    .arg("-vcodec")
    .arg("mpeg4")
    .arg("-f")
    .arg("mpegts")
    .arg(format!("udp://{}", ip))
    .spawn();

    println!("Streaming");

    return child;
}


// pub fn save_stream_to_file(stream: &str, filename: &str) -> Child{
    
//     let stream = "videoplayback.mp4";

//     let filename = filename;

//     let mut child = Command::new("ffmpeg")
//     .arg("-re")
//     .arg("-i")
//     .arg(stream)
//     .arg("-vf")
//     .arg("fps=10/1")
//     .arg("-update")
//     .arg("1")
//     .arg("output.png")
//     .arg("-y")
//     .spawn()
//     .unwrap();

//     return child;

//     // std::thread::sleep(std::time::Duration::from_secs(20));

//     // child.kill().unwrap();

// }



use image::DynamicImage;


//create a macro that will return None if the value cannot be unwrapped
macro_rules! try_none {
    ($e:expr) => (match $e { Ok(e) => e, Err(_) => return None })
}


pub fn file_to_image(path: &str, dimensions: (u32,u32)) -> Option<DynamicImage> {
    let img = try_none!(image::open(path));
    let img = img.resize_exact(dimensions.0, dimensions.1, image::imageops::FilterType::Nearest);
    Some(img)
}






pub fn image_to_tensor_vector( image1: DynamicImage ) -> Vec<Vec<Vec<f32>>>{

    let image1 = image1.to_rgb16();

    let mut r = Vec::new();
    let mut g = Vec::new();
    let mut b = Vec::new();

    for pixelrow in image1.rows(){

        let mut rowr = Vec::new();
        let mut rowg = Vec::new();
        let mut rowb = Vec::new();

        for pixel in pixelrow{

            rowr.push(pixel[0] as f32 / 65536.0);
            rowg.push(pixel[1] as f32 / 65536.0);
            rowb.push(pixel[2] as f32 / 65536.0);
        }

        r.push(rowr);
        g.push(rowg);
        b.push(rowb);
    }

    //println!("{:?}, {:?}, {:?}", r.len(), r[0].len(), r[0].len());

    return vec![r, g, b];
}



pub fn image_similarity( image1: DynamicImage, image2: DynamicImage) -> f32{

    //scale both to 20 x 20
    let image1 = image1.resize_exact(6, 4, image::imageops::FilterType::Nearest);
    let image2 = image2.resize_exact(6, 4, image::imageops::FilterType::Nearest);

    let tensorvector1 = image_to_tensor_vector(image1);
    let tensorvector1 = tensorvector1.into_iter().flatten().flatten().collect::<Vec<f32>>();

    let tensorvector2 = image_to_tensor_vector(image2);
    let tensorvector2 = tensorvector2.into_iter().flatten().flatten().collect::<Vec<f32>>();

    let mut sum = 0.0;

    for i in 0..tensorvector1.len(){
        sum += 1.0 - (tensorvector1[i] - tensorvector2[i]).abs();
    }

    return sum / tensorvector1.len() as f32;

}



pub fn save_stream_to_file(stream: &str, filename: &str) -> Child{
    
    // let stream = "videoplayback.mp4";

    let filename = filename;

    let mut child = Command::new("ffmpeg")
    .arg("-re")
    .arg("-i")
    .arg(stream)
    .arg("-vf")
    .arg("fps=1/1")
    .arg("-update")
    .arg("1")
    .arg( filename )
    .arg("-y")
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::null())
    .spawn()
    .unwrap();

    return child;

}





pub fn save_stream( source: &str, output: &str ) -> Child{

    //ffmpeg -i input.mp4 -vf fps=1 screenshot-%03d.png

    let mut child = Command::new("ffmpeg")
    .arg("-i")
    .arg(source)
    .arg("-vf")
    .arg("fps=2/1")
    .arg( output.to_string()+"%05d.jpg" )
    //.stdout(std::process::Stdio::null())
    //.stderr(std::process::Stdio::null())
    .spawn()
    .unwrap();

    return child;

}

