from diffusers import StableDiffusionXLPipeline
import hashlib
import torch

import os
from os.path import join


def generate_md5(filename:str) -> str:
    '''
    Generate md5 hash of a file
    Input:
        - filename (str): path to the file
    '''
    # Open the file in binary mode
    with open(filename, 'rb') as f:
        # Read the file
        data = f.read()
        # Generate the MD5 hash
        md5_hash = hashlib.md5(data).hexdigest()
    return md5_hash


class DataGen():
    def __init__(self, model_path:str, batch_size:int=5, save_dir:str='./data'):
        '''
        Data Gen for ADL Project
        Args:
            - model_path (str): path to the model
            - batch_size (int): batch size for the model to generate
        '''
        self.model_path = model_path
        self.pipeline = StableDiffusionXLPipeline.from_single_file(self.model_path, use_safetensors=True, torch_dtype=torch.float16).to("cuda")
        self.batch_size = batch_size
        self.save_dir = save_dir

    def rename_file(self, file_path:str):
        '''
        Rename a file to the md5sum of the file
        Args:
            - file_path (str): path to the file
        '''
        h = generate_md5(file_path)
        new_name = join(self.save_dir, f'./{h}.png')
        os.rename(file_path, new_name)

    def generate(self, prompt:str):
        '''
        Generate images from text
        Input:
            - prompt (str): text to generate images from
        Returns:
            - images (list): list of images generated from the text
        '''
        images = self.pipeline(prompt, num_images_per_prompt=self.batch_size).images
        for img in images:
            img.save('./test.png')
            self.rename_file('./test.png')
        del(images)
        return

if __name__ == '__main__':
    base_model = '/home/squirt/Documents/weights/anime_weights/others/bluePencilXL_v310.safetensors' 
    gen = DataGen(base_model)

    with open('prompts.txt', 'r') as f:
        lines = f.read().splitlines()

    lines = list(filter(lambda x: x != '', lines))
    
    for l in lines:
        for i in range(10):
            gen.generate(l)

    print('Done')


