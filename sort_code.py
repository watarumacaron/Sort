import os
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from StyleGAN import StyleGANGenerator as G
import lpips

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlatents_path_4_sort', type=str, help='Paht of the dlatents to sort.')
    parser.add_argument('--example_path', type=str, help='Example to sort dlatents.')
    parser.add_argument ('--output_dir', type=str)
    parser.add_argument('--dir4img', type=str)
    return parser.parse_args()

def main():
    args = parse_args()

    loss_fn = lpips.LPIPS(net='vgg', spatial=False)
    if torch.cuda.is_available():
        loss_fn = loss_fn.to('cuda')

    dlatents = np.load(args.dlatents_path_4_sort)
    example = np.load(args.example_path)

    generator = G()

    '''Calculate LPIPS loss and return the index with the lowest loss value.'''
    id_ls = []
    for i in range(dlatents.shape[0]):
        image4sort = generator.synthesis(dlatents[i])
        loss_ls = []
        for j in range(example.shape[0]):
            example_image = generator.synthesis(example[j])
            lpips_loss = loss_fn.forward(image4sort, example_image)
            loss_ls.append(lpips_loss.item())
        loss_ls = np.array(loss_ls)
        id_ls.append(np.argmin(loss_ls))

    
    '''Convert to dictionary type.'''
    final = {}
    for j in range(example.shape[0]):
      match = []
      for i, index in enumerate(id_ls):
        if index == j:
          match.append(i)
      final[f'{j}'] = match
    
            

    '''If the length of the value of final is greater than or equal to 2, recalculate and return the smallest index.'''
    id_ls = {}
    for key, value in final.items():
      if len(value) > 1:
        loss_ls = []
        example_image = generator.synthesis(example[int(key)])
        for i in range(len(value)):
          image4sort = generator.synthesis(dlatents[value[i]])
          lpips_loss = loss_fn.forward(image4sort, example_image)
          loss_ls.append(lpips_loss.item())
        loss_ls = np.array(loss_ls)
        final[key] = value[np.argmin(loss_ls)]

    for key, value in final.items():
      for key2, value2 in id_ls.items():
          if key == key2:
            final[key] = value2

    for key, value in final.items():
      if type(value) == list:
        final[key] = value[0]

    '''Create sorted dlatents.'''
    for key, value in final.items():
      if int(key) == 0:
        sorted_dlatents = dlatents[value][None]
      else:
        sorted_dlatents = np.append(sorted_dlatents, dlatents[value][None], axis=0)

    print("Save sorted dlatents.")
    np.save(args.output_dir, sorted_dlatents)

    print("----------------------------------------")

    folders = os.listdir('/content/drive/MyDrive/今西/SortedInvertedCode/images')
    if len(folders) > 0:
      for i, name in enumerate(folders):
        num = int(name[len('images'):])
        if i == 0:
          max_num = num
        else:
          if num > max_num:
            max_num = num
    else:
      max_num = 0
    folder_path = args.dir4img + f'/images{max_num + 1}'
    os.makedirs(folder_path)

    bar = '-'
    for i in range(sorted_dlatents.shape[0]):
        image = generator.synthesis(sorted_dlatents[i])
        image = generator.process4imshow(image)
        image = Image.fromarray(image)
        image_path = folder_path + f'/{i}.jpg'
        image.save(image_path)
        print(f'{i+1}/{sorted_dlatents.shape[0]}:{bar}')
        bar += '-'

    print("Finish.")


if __name__ == "__main__":
    main()
