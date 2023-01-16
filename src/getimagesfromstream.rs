use std::path::PathBuf;
use std::sync::mpsc::channel;
use rand::Rng;

use image::{DynamicImage, GrayImage, RgbImage, GenericImageView};

use image::Pixels;
use image::Rgb;
use std::collections::HashMap;


use std::io::Write;
use std::process::Command;
use std::process::Child;
use std::io::Error;

use crate::save_stream_to_file;
use crate::file_to_image;

pub struct ImageStream{
    filename: String,
    screenshotprocess: Child,
    lastimage: DynamicImage,
}

impl ImageStream{

    pub fn new(ip: &str) -> ImageStream{

        //create a random string with length 4
        let mut rng = rand::thread_rng();
        let filename: String = (0..5).map(|_|  rng.sample(rand::distributions::Alphanumeric)  as char ).collect::<String>() + ".png";

        let mut child = save_stream_to_file(ip, &filename);
        ImageStream{
            filename: filename.to_string(),
            screenshotprocess: child,
            lastimage: DynamicImage::new_rgb16(1,1),
        }
    }

    pub fn poll(&mut self) -> Option<DynamicImage>{
        if let Some(image) = file_to_image( &self.filename, crate::DIMENSIONS ){
            if image != self.lastimage{
                self.lastimage = image.clone();
                return Some(image);
            }
        }
        return None;
    }


}


impl Drop for ImageStream{
    fn drop(&mut self) {
        self.screenshotprocess.kill().unwrap();
        //std::fs::remove_file(&self.filename).unwrap();
    }
}

//save the stream to a 
