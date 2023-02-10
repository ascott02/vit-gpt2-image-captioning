import io
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
os.chdir(parent)

# for web.py
import web
import re
import config
import base64
import datetime

urls = (
    '/', 'index',
    '/api', 'api',
    '/login', 'login',
    '/upload', 'upload',
    '/batch', 'batch',
)

import logging
log = logging.getLogger(config.log_file)
if not len(log.handlers):
    log.setLevel(logging.INFO)
    loghandler = logging.FileHandler(config.log_file)
    log.addHandler(loghandler)

import json
import yaml
import torch
import requests
import numpy as np
import gc
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



def uri_validator(x):
    try:
        result = requests.urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        if x.startswith("http"):
            return True
        else:
            return False

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    try:
        i_image = Image.open(image_path)
    except:
        stream = io.BytesIO(image_path)
        i_image = Image.open(stream)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


def get_caption(image_batch):

    for i,image in enumerate(image_batch):
        if uri_validator(image):
            try:
                response = requests.get(image, stream=True)
            except Exception as e:
                print(f"Exception occurred retrieving image {image}: {str(e)}")
                continue
            if response.status_code == 200:
                # set decod content value to True, otherwise the downloaded image file's size will be zero
                response.raw.decode_content = True
                image_batch[i] = response.raw
            else:
                print(f"Image {image} could not be retrieved. Status code {response.status_code} returned.")
                continue
    
    caption = predict_step(image_batch)
    return caption


class index:
    def GET(self, *args):
        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return """<html><head></head><body>
This form takes an image upload returns caption.<br/><br/>
It uses vit-gpt-2 from huggingface <a href="https://huggingface.co/nlpconnect/vit-gpt2-image-captioning">https://huggingface.co/nlpconnect/vit-gpt2-image-captioning</a>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
<br/><br/>
<input type="submit" />
</form>
</body></html>"""
        else:
            raise web.seeother('/login')

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['img_file'].filename)    # This is the filename

        caption = get_caption([x['img_file'].value])

        # tokens = demo.predict(x['img_file'].value)
        # caption = demo.caption_processor(tokens.tolist()[0])['caption']

        data_uri = base64.b64encode(x['img_file'].value)
        img_tag = '<img src="data:image/jpeg;base64,{0}">'.format(data_uri.decode())

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /index img_file: {x['img_file'].filename}")

        page = """<html><head></head><body>
This form takes an image upload and caption and returns an IICR rating.<br/><br/>
It uses vit-gpt-2 from huggingface <a href="https://huggingface.co/nlpconnect/vit-gpt2-image-captioning">https://huggingface.co/nlpconnect/vit-gpt2-image-captioning</a>
<form method="POST" enctype="multipart/form-data" action="">
Image: <input type="file" name="img_file" /><br/><br/>
<br/><br/>
<input type="submit" />
</form>""" + img_tag + """<br/>Caption: """ + caption[0] + """<br/>
</body></html>"""

        if web.ctx.env.get('HTTP_AUTHORIZATION') is not None:
            return page
        else:
            raise web.seeother('/login')


class api:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['img_url'])                # This is the URL to the image

        if not x['img_url']:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."
    

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /api token: {x['token']}, img_url: {x['img_url']}")
        caption = get_caption([x['img_url']])
        return caption


class batch:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        web.debug(x['json_payload'])           # This is the URL to the image

        if not x['json_payload']:
            return "No file."

        if not x['token']:
            return "No token."

        if not x['token'] in config.tokens:
            return "Not in tokens."

        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /batch token: {x['token']}, json_payload: {x['json_payload']}")
        json_dict = json.loads(x['json_payload'])
        captions = {}
        for image in json_dict['images']:
            # caption = get_caption(json_dict['images'])
            captions[image] = get_caption([image])
        # return [x['caption'] for x in captions]
        return json.dumps(captions)
        # captions = get_caption(json_dict['images'])
        # return captions


class upload:

    def POST(self, *args):
        x = web.input(img_file={})
        web.debug(x['token'])                  # This is the api token 
        # web.debug(x['img_data'].filename)      # This is the filename

        if not x['img_data']:
            return "No file."

        if not x['token']:
            return "No token."
        token = x['token'].decode()

        if not token in config.tokens:
            return "Not in tokens."
    
        ip = web.ctx.ip
        now = datetime.datetime.now()
        log.info(f"{now} {ip} /upload token: {token}")
        caption = get_caption([x['img_data']])
        return caption


class login:

    def GET(self):
        auth = web.ctx.env.get('HTTP_AUTHORIZATION')
        authreq = False
        if auth is None:
            authreq = True
        else:
            auth = re.sub('^Basic ','',auth)
            print("DEBUG auth:", auth)
            # username,password = base64.decodestring(auth).split(':')
            username,password = base64.b64decode(auth).decode().split(':')
            if (username,password) in config.allowed:
                raise web.seeother('/')
            else:
                authreq = True
        if authreq:
            web.header('WWW-Authenticate','Basic realm="Auth example"')
            web.ctx.status = '401 Unauthorized'
            return

app = web.application(urls, globals(), autoreload=False)

if __name__ == "__main__":
    app.run()
