from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os
from scipy.misc import imresize
import argparse

imagestore=[]


parser=argparse.ArgumentParser(description='Source Video path')
parser.add_argument('source_vid_path',type=str)
parser.add_argument('fps',type=int)
args=parser.parse_args()

video_source_path= args.source_vid_path
fps=args.fps

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)
	img=imresize(img,(227,227,3))
	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
	imagestore.append(gray)



videos=os.listdir(video_source_path)
print("Found ",len(videos)," training video")


create_dir(video_source_path+'/frames')

remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'

for video in videos:
		os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
		images=os.listdir(framepath)
		for image in images:
			image_path=framepath+ '/'+ image
			store(image_path)


imagestore=np.array(imagestore)
a,b,c=imagestore.shape
imagestore.resize(b,c,a)
imagestore=(imagestore-imagestore.mean())/(imagestore.std())
imagestore=np.clip(imagestore,0,1)
np.save('training.npy',imagestore)
os.system('rm -r {}'.format(framepath))