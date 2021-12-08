import os

root_dir = '/home/chris/src/python/tensorflow-yolov4-tflite/'
# root_dir = '/media/fast/climberg/src/python/tensorflow-yolov4-tflite/'



imgs = [
os.path.join(root_dir,'visualization/imgs/city-438393_1920.jpg'),
os.path.join(root_dir,'visualization/imgs/city-731239_1920.jpg'),
os.path.join(root_dir,'visualization/imgs/horses-1414889_1280.jpg'),
os.path.join(root_dir,'visualization/imgs/kitten-316995_1280.jpg'),
os.path.join(root_dir,'visualization/imgs/pizza-2776188_1280.jpg'),
os.path.join(root_dir,'visualization/imgs/uber-eats-4709288_1920.jpg'),
        ]


out_path_bboxes = os.path.join(root_dir,'visualization/out_bb')
out_path_shift = os.path.join(root_dir,'visualization/out_shift')
out_path_optimization = os.path.join(root_dir,'visualization/out_opti')