import glob
import argparse
from PIL import Image
from tqdm import tqdm
import os
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", help = "Directory of images to convert")
    parser.add_argument("-e", help = "Extension of images to convert")
    parser.add_argument("-fps", help = "Frames per second of output video")
    parser.add_argument("-n", help = "Name of output video")


    args = parser.parse_args()

   

    if(args.f == None):
        print("\nPlease specify a directory to read with -f")
        exit()
    else:
        image_folder = str(args.f)
        print(f"\nLoading Images from {image_folder} ...")
    


    video_name = str(args.n) if args.n != None else 'video.mp4'
    extension = str(args.e) if args.e != None else 'png'
    fps = int(args.fps) if args.fps != None else 10
    #fourcc = cv2.CV_FOURCC(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    images = [img for img in sorted(os.listdir(image_folder), reverse=False) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    
    for image in tqdm(images, desc='Creating Video'):
        video.write(cv2.imread(os.path.join(image_folder, image))) 
   
    video.release() 

