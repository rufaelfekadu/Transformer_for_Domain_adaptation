import os
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# ... (other functions and imports)

def create_dirs(datasetDir):
    class_names = ['bicycle', 'car', 'person']
    for cls_name in class_names:
        if not os.path.exists(os.path.join(datasetDir, f'sgada_data_thread/mscoco/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'sgada_data_thread/mscoco/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'sgada_data_thread/mscoco/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'sgada_data_thread/mscoco/val/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'sgada_data_thread/flir/train/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'sgada_data_thread/flir/train/{cls_name}/'))
        if not os.path.exists(os.path.join(datasetDir, f'sgada_data_thread/flir/val/{cls_name}/')):
            os.makedirs(os.path.join(datasetDir, f'sgada_data_thread/flir/val/{cls_name}/'))

def get_id(dataset, set_type, i):
    if dataset=='flir':
        if set_type=='train':
            return i+1, 5 - len(str(i))
        else:
            ID = i+8863
            if (len(str(ID))<5):
                zeros = 5 - len(str(ID))
            elif (len(str(ID))==5):
                zeros = -1

            return ID, zeros
    else:
        return i, 12 - len(str(i))

def parse_mscoco_threaded(dataset, datasetDir, annotations, set_type='train', num_threads=4):
    if dataset =='mscoco':
        path_to_images = f'mscoco/{set_type}2017'
    else:
        path_to_images = f'FLIR_ADAS_1_3/{set_type}/thermal_8_bit'
    c1 = 0
    count = 0
    im_crop = [0, 0, 0, 0]
    ann = pd.DataFrame(columns=['path', 'label'])
    image_data_to_save = []
    def process_annotation(i):
        nonlocal c1, count, ann
        cat_map = ['person', 'bicycle', 'car']
        if i['category_id'] in [1, 2, 3]:  # Only process specified categories
            c1 = c1 + 1
            # ID = i['image_id']
            # zeros = 12 - len(str(ID))
            ID, zeros = get_id(dataset,set_type,i['image_id'])
            zero = '0' * zeros

            if dataset =='mscoco':
                image_path = os.path.join(datasetDir, '{}/{}{}.jpg'.format(path_to_images, zero, ID))
            else:
                image_path = os.path.join(datasetDir, '{}/FLIR_{}{}.jpeg'.format(path_to_images, zero, ID))
            im = Image.open(image_path)
            bbox = i['bbox']
            im_crop[0] = i['bbox'][0]
            im_crop[1] = i['bbox'][1]
            height = im.size[1]
            width = im.size[0]
            # ... (rest of the processing logic)
            if i['bbox'][2] > i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[1] = i['bbox'][1] - ((i['bbox'][2]-i['bbox'][3])/2)
            if i['bbox'][3] > i['bbox'][2]:
                im_crop[2] = i['bbox'][0] + ((i['bbox'][2]+i['bbox'][3])/2)
                im_crop[0] = i['bbox'][0] - ((i['bbox'][3]-i['bbox'][2])/2)
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if i['bbox'][2] == i['bbox'][3]:
                im_crop[2] = i['bbox'][0] + i['bbox'][2]
                im_crop[3] = i['bbox'][1] + i['bbox'][3]
            if (im_crop[2] > width) or (im_crop[3]  > height) or (im_crop[0]<0) or (im_crop[1]<0):
                count = count + 1
                
            else:
                # area = im.crop(im_crop)
                image_data_to_save.append((im.crop(im_crop),'{}/{}/{}/{}.jpg'.format(dataset,set_type, cat_map[i['category_id'] - 1], c1), i['category_id'] - 1))
                # area.save(os.path.join(datasetDir, 'sgada_data_thread/{}/{}/{}/{}.jpg'.format(dataset,set_type, cat_map[i['category_id'] - 1], c1)))

            # Instead of saving the image and appending to the DataFrame, you can return the relevant data
            # return ['{}/{}/{}/{}.jpg'.format(dataset,set_type, cat_map[i['category_id'] - 1], c1), i['category_id'] - 1]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm(executor.map(process_annotation, annotations['annotations']), total=len(annotations['annotations'])))

    # def save_image(img_file):
        # img, path = img_file
        # img.save(os.path.join(datasetDir, 'sgada_data_thread/'+path))
    
    # Filter out any None results (failed processing)
    # results = [r for r in results if r is not None]
    for img, path, label in tqdm( image_data_to_save):
        img.save(os.path.join(datasetDir, 'sgada_data_thread/'+path))
        ann = ann.append(pd.DataFrame([[path, label]], columns=['path', 'label']), ignore_index=True)
    # ann = ann.append(pd.DataFrame(results, columns=['path', 'label']), ignore_index=True)
    
    # print(ann.head())
    # ann.to_csv(os.path.join(datasetDir, 'sgada_data_thread/mscoco_{}.txt'.format(set_type)), header=None, index=None, sep=' ', mode='a')

# ... (other functions)

def main():
    datasetDir = os.environ['DATASETDIR']

    print(f'Creating output dirs if not exist under {datasetDir}')
    create_dirs(datasetDir)

    print('Loading MSCOCO training set annotations')
    with open(os.path.join(datasetDir, 'mscoco/annotations_trainval2017/annotations/instances_train2017.json'),'r') as f:
        data = json.load(f)
    print('Parsing MSCOCO training set')
    parse_mscoco_threaded('mscoco', datasetDir, data, set_type='train', num_threads=128)  # Adjust the number of threads as needed

    # print('Loading MSCOCO validation set annotations')
    # with open(os.path.join('mscoco',datasetDir, 'mscoco/annotations_trainval2017/annotations/instances_val2017.json'),'r') as f:
    #     data = json.load(f)
    # print('Parsing MSCOCO validation set')
    # parse_mscoco_threaded('mscoco',datasetDir, data, set_type='val', num_threads=256)  # Adjust the number of threads as needed

    print('Loading FLIR training set annotations')
    with open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/train/thermal_annotations.json'),'r') as f:
        data = json.load(f)
    print('Parsing FLIR training set')
    parse_mscoco_threaded('flir', datasetDir, data, set_type='train', num_threads=128)  # Adjust the number of threads as needed

    print('Loading FLIR validation set annotations')
    with open(os.path.join(datasetDir, 'FLIR_ADAS_1_3/val/thermal_annotations.json'),'r') as f:
        data = json.load(f)
    print('Parsing FLIR validation set')
    parse_mscoco_threaded('flir', datasetDir, data, set_type='val', num_threads=128)  # Adjust the number of threads as needed

    # ... (other function calls)

    

if __name__ == '__main__':
    main()
