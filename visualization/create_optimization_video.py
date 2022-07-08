#!/usr/bin/env python3

from visualization.common import get_subdirectories, setup_clean_directory
import os


out_path = '/home/chris/exp5'
img_path = '/home/chris/exp5/exp1'
tmp_path = '/home/chris/exp5/tmp'

setup_clean_directory(tmp_path)



cls = get_subdirectories(img_path)

num_steps = 15000
step_size = 250
final_steps = 30

global_i = 0

for c in cls:
    final_img = os.path.join(img_path,c,'final.jpg')
    final_img_prelabel = os.path.join(img_path,c,'final_prelabeled.jpg')
    final_img_label = os.path.join(img_path,c,'final_labeled.jpg')
    command_prelabel = 'width=`identify -format %w '+final_img+'`; convert -background "#0008" -fill white -gravity center -size ${width}x30 caption:"'+c+'!" '+final_img+' +swap -gravity south -composite  '+final_img_label
    command_label = 'width=`identify -format %w '+final_img+'`; convert -background "#0008" -fill white -gravity center -size ${width}x30 caption:"YOLO thinks, this is a..." '+final_img+' +swap -gravity south -composite  '+final_img_prelabel

    if not os.path.exists(final_img):
        print(c+' IS INCOMPLETE, WILL BE OMMITED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        continue
    for r in range(0,num_steps,step_size):
        img_name = os.path.join(img_path,c,"%d.jpg"%(r,))
        os.system("cp "+img_name+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1
    os.system(command_prelabel)
    os.system(command_label)
    for r in range(final_steps):
        os.system("cp "+final_img_prelabel+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1

    for r in range(final_steps):
        os.system("cp "+final_img_label+" "+os.path.join(tmp_path,'%08d.jpg'%(global_i,)))
        global_i += 1


render_command = 'ffmpeg -r 15 -i '+os.path.join(tmp_path,'%08d.jpg')+' -c:v libx264 -vf "fps=25,format=yuv420p" '+os.path.join(out_path,'out.mp4')
os.system(render_command)
