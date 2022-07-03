from visualization.common import get_subdirectories, setup_clean_directory
import os


img_path = '/media/fast/climberg/src/python/tensorflow-yolov4-tflite/visualization/opt_imgs/imgs/'
tmp_path = '/media/fast/climberg/src/python/tensorflow-yolov4-tflite/visualization/opt_imgs/tmp/'

setup_clean_directory(tmp_path)



cls = get_subdirectories(img_path)

num_steps = 15000
step_size = 250
final_steps = 30

global_i = 0

for c in cls:
    for r in range(0,num_steps,step_size):
        img_name = os.path.join(img_path,c,"%d.jpg"%(r,))
        os.system("cp "+img_name+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1
    final_img = os.path.join(img_path,c,'final.jpg')
    final_img_prelabel = os.path.join(img_path,c,'final_prelabeled.jpg')
    final_img_label = os.path.join(img_path,c,'final_labeled.jpg')
    command_prelabel = 'width=`identify -format %w '+final_img+'`; convert -background "#0008" -fill white -gravity center -size ${width}x30 caption:"'+c+'!" '+final_img+' +swap -gravity south -composite  '+final_img_label
    command_label = 'width=`identify -format %w '+final_img+'`; convert -background "#0008" -fill white -gravity center -size ${width}x30 caption:"YOLO thinks, this is a..." '+final_img+' +swap -gravity south -composite  '+final_img_prelabel
    os.system(command_prelabel)
    os.system(command_label)
    for r in range(final_steps):
        os.system("cp "+final_img_prelabel+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1

    for r in range(final_steps):
        os.system("cp "+final_img_label+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1


# ffmpeg -r 15 -i %08d.jpg -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4


