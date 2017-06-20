import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_json_information(json_file):
    '''
    
    :param json_file: path of json files
    :return: 
      result format : [{'filename':'*.png', 'bbox_x1': value, 'bbox_y1':value, 'bbox_width':value, 'bbox_height': value,
      'boxes':np.array([{'left':value, 'top':value, 'width':value, 'height':value}, ..]), labels:np.array([val1, val2,..])}, ...]
      'filename' : name of image file
      'bbox_x1' : left coordinate of bounding box
      'bbox_y1' : top coordinate of bounding box
      'bbox_width : width of bounding box
      'bbox_height' : height of bounding box
      'boxes' : all box in the image
      'boxes'.'left': left coordinate of one box
      'boxes'.'top' : top coordinate of one box
      'boxes'.'widht' : width of one box
      'boxes'.'height' : height of one box
    '''
    if not os.path.exists(json_file):
        raise Exception("{0} is not exist".format(json_file))

    with open(json_file, 'r') as file:
        json_info = json.load(file)
    results = []
    for info in json_info:
        digit_info = {}
        digit_info['filename'] = info['filename']

        boxes = info['boxes']
        x1 = int(np.min([box['left'] for box in boxes]))
        y1 = int(np.min([box['top'] for box in boxes]))
        x2 = int(np.max([box['left'] + box['width'] for box in boxes]))
        y2 = int(np.max([box['top'] + box['height'] for box in boxes]))

        digit_info['bbox_x1'] = x1
        digit_info['bbox_y1'] = y1
        digit_info['boxes'] = np.array([{ name : int(box[name]) for name in ['left', 'top', 'width', 'height']}  for box in boxes])
        digit_info['bbox_width'] = x2 - x1
        digit_info['bbox_height'] = y2 - y1
        digit_info['labels'] = np.array([int(box['label']) for box in boxes])
        results.append(digit_info)
    return results

jsons_root = './jsons'

train_json_file = os.path.join(jsons_root, 'train/digitStruct.json')
test_json_file = os.path.join(jsons_root, 'test/digitStruct.json')
train_json_info = get_json_information(train_json_file)
test_json_info = get_json_information(test_json_file)
print("Train json info : ", train_json_info[0:5])
print("Test json info : ", test_json_info[0:5])

print("=" * 100)
train_pd = pd.DataFrame(train_json_info)
print(train_pd.head())

valid_train_info = train_pd.loc[train_pd.bbox_height >= 20, :]
train_info = np.array(train_json_info)[valid_train_info.index]

dest_img_width = 64
dest_img_height = 64


def calculate_crop_coor( digit_center, square_center, digit_length, img_length, extend_ratio):
    # image length after entend
    digit_length_ext = int(digit_length * extend_ratio)
    digit_half_length_ext = digit_length_ext // 2

    img_crop_floor_coor = np.max([ 0, digit_center - digit_half_length_ext])
    img_crop_ceil_coor = np.min([ img_length, digit_center + digit_half_length_ext])
    square_crop_floor_coor = square_center - (digit_center - img_crop_floor_coor)
    square_crop_ceil_coor = square_center + (img_crop_ceil_coor - digit_center)
    return (img_crop_floor_coor, img_crop_ceil_coor), (square_crop_floor_coor, square_crop_ceil_coor)

def get_img_coordinate_in_square(one_img_info, img_width, img_height, extend_ratio):
    # x1 coordinate in image
    digit_x1 = one_img_info["bbox_x1"]
    digit_y1 = one_img_info["bbox_y1"]
    digit_width = one_img_info["bbox_width"]
    digit_height = one_img_info["bbox_height"]
    print('digit_x1 : ', digit_x1)
    print('digit_y1 : ', digit_y1)
    print('digit_width : ', digit_width)
    print('digit_height : ', digit_height)
    digit_center_x = digit_x1 + digit_width // 2
    digit_center_y = digit_y1 + digit_height // 2
    # length of output square area
    square_length = int(np.max([digit_width, digit_height]) * extend_ratio)
    # x and y coordinate of square area is the same
    square_center_coor = square_length // 2
    img_x_coor, square_x_coor = calculate_crop_coor( digit_center_x, square_center_coor, digit_width, img_width, extend_ratio)
    img_y_coor, square_y_coor = calculate_crop_coor( digit_center_y, square_center_coor, digit_height, img_height, extend_ratio)
    return square_length, (img_x_coor[0], img_y_coor[0], img_x_coor[1], img_y_coor[1]), \
           (square_x_coor[0], square_y_coor[0], square_x_coor[1], square_y_coor[1])

test = {'filename': '1.png', 'bbox_x1': 246, 'bbox_y1': 77, 'bbox_width': 173, 'bbox_height': 223}
print(get_img_coordinate_in_square(test, 350, 300, 1.2))

# extend ratio:
extend_ratio = 1.2
def process_one_image( one_img_info, img_dir, show = False):
    img_path = os.path.join( img_dir, one_img_info['filename'])
    print("Process image : ", img_path)
    if not os.path.isfile(img_path):
        raise Exception( img_path, " is not exist")
    from scipy import ndimage
    img_data = ndimage.imread(img_path, flatten=True).astype(np.float)
    img_height, img_width = img_data.shape[0], img_data.shape[1]
    square_length, img_crop_coor, square_fill_coor = get_img_coordinate_in_square(one_img_info, img_width, img_height, extend_ratio)
    square_output = np.ones([square_length, square_length]) * np.mean(img_data)

    # copy image to square area
    print("img_crop_coor : ", img_crop_coor)
    print("square_fill_coor : ", square_fill_coor)

    square_output[square_fill_coor[1]: square_fill_coor[3], square_fill_coor[0] : square_fill_coor[2]] = \
        img_data[img_crop_coor[1]:img_crop_coor[3], img_crop_coor[0]:img_crop_coor[2]]
    if show:
        plt.subplot(2,2,1)
        plt.imshow(img_data)
        plt.subplot(2,2,2)
        plt.imshow(square_output)
    zoom_factor = 1.0 * dest_img_height / square_length
    img_zoomed = ndimage.interpolation.zoom(square_output, zoom_factor)
    img_return = img_zoomed.astype(np.float32) - np.min(square_output)
    img_return = img_return / np.max(img_return) - 0.5
    if show:
        plt.subplot(2,2,3)
        plt.imshow(img_return)
        plt.show()

    return img_return, one_img_info['labels']

#img_return, label = process_one_image(train_json_info[0], './data/train', True)
#print("img_return shape : ", img_return.shape)

from multiprocessing import Pool
from functools import partial

def process_images(img_list, img_path):
    pool = Pool()
    results = pool.map(partial(process_one_image, **{'img_dir':img_path}), img_list)
    pool.close()
    pool.join()
    imgs, labels = zip(*results)
    return np.array(imgs), np.array(labels)

print("Processing images")
np.random.shuffle(train_json_info)
img_list, label_list = process_images(train_json_info, "./data/train")
print("img_list : ", img_list.shape)
print("label_list : ", label_list.shape)
