import json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# Open the JSON file

image_path = '../test/tii/RGB' 
out_dir = '../test/tii/daytime'
out_label = '../test/tii/rgb.txt'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open('../test/tii/daytime.json', 'r') as json_file:

    with open(out_label, 'w') as output_file:
        # Load the JSON data
        data = json.load(json_file)
        # Loop through each image in the JSON data
        img = []
        i = 0
        for image in tqdm(data['annotations'][:500]):
            catid = image['category_id']
            label_dir = os.path.join(out_dir,data['categories'][catid-1]['name'])

            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            imgid = image['image_id']
            img.append(imgid)

            image_filename = data['images'][imgid-1]['file_name']
            im = Image.open(os.path.join(image_path,image_filename))
            
            # Get the bounding box coordinates
            x_min = image['bbox'][0]
            y_min = image['bbox'][1]
            x_max = image['bbox'][0]+(image['bbox'][2])
            y_max = image['bbox'][1]+(image['bbox'][3])

            # Crop out the bounding box region
            bbox_im = im.crop((x_min, y_min, x_max, y_max))
            # Get the bounding box label
            bbox_label = image['category_id']
            new_image_path = os.path.join(label_dir,'{}.png'.format(i) )
            # Save the cropped image with the bounding box label as the filename
            bbox_im.save(new_image_path)
            output_file.write('{}/{} {}\n'.format(data['categories'][catid-1]['name'], '{}.png'.format(i), bbox_label-1))
            i+=1