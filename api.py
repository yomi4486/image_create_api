# FastAPIの読み込み
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import RedirectResponse,FileResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import json
from os.path import join, dirname
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import StableDiffusionPipeline,AutoencoderKL
import torch
from torch import autocast

load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

hugging_token = os.environ.get("HUGGING_TOKEN")
login(hugging_token)

if not os.path.exists("./result"):
    os.mkdir('./result')

print('SD2.1の読み込み...')
# 画像生成に使うライブラリやその設定などを指定いいいいいいいいいいいいいいいいいいいいいｗｗｗ
ldm = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=hugging_token,
    low_cpu_mem_usage=True,
).to(torch.device('cuda:0'))

print('イラスト用モデルの読み込み...')
illust_pipe = StableDiffusionPipeline.from_single_file(
    os.environ.get("MODEL_PATH"),
    vae=AutoencoderKL.from_single_file("./model/vae/Counterfeit-V2.5.vae.pt"),
    load_safety_checker=False,

).to(torch.device('cuda:0'))

async def create_image(prompt,filename,create_mode):
    negative_quality_tags ="blurry,lowres,duplicate,morbid,deformed,monochrome,greyscale,comic,4koma,2koma,sepia,simple background,rough,unfinished,horror,duplicate legs,duplicate arms,error,worst quality,normal quality,low quality,ugly,bad anatomy,out of focus,jpeg artifacts,text,missing limb,missing fingers,bad hands,extra digit,fewer digits,signature,pubic hair,long_body,long_neck,longbody,missing arms,poorly_drawn_hands,malformed_hands,deformed eyes,cropped,freckles,multiple people,multipul angle,split view,grid view,Strong background,inaccurate limb,floating_limbs,disconnected_limbs,liquid fingers,malformed hair,duplicate hair,ugly face,bad face,open mouth,flat shading,flat color,unusually long hair,3D,distorted eyes,distorted face shape,missing body,distant eyes,bokeh,half-open eyes,asymmetrical"

    with autocast("cuda"):
        if create_mode == 1:
            image = illust_pipe(prompt,
                    guidance_scale=11,
                    num_inference_steps=60,
                    batch_size=4,
                    negative_prompt=negative_quality_tags,
                    cfg_scale=2, #cfg_scaleのデフォルトの値は4
                    mask_by=-4,
                    strength=1,
                    size=(512,512)
                    ).images[0]

        elif create_mode == 0:
            image = ldm(prompt,
                    num_inference_steps=56,
                    batch_size=1,
                    negative_prompt=negative_quality_tags,
                    cfg_scale=2, #cfg_scaleのデフォルトの値は4
                    mask_by=-4,
                    strength=1,
                    size=(512,512)
                    ).images[0] 

    image.save(f".\\result\\{filename}.png")

# FastAPIのインスタンスを作成
app = FastAPI(title="USB無くしたEngine",version="1.0.0",description="画像生成を行えるAPIです。")

ip_last_access = {}

# Middleware to handle CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    print(request.url)
    client_ip = request.client.host

    # Exclude requests from 127.0.0.1 (localhost)
    if not str(request.url) in ["/openapi.json","/docs"] or (client_ip != "127.0.0.1"):

        # Get the last access timestamp for this IP
        last_access_time = ip_last_access.get(client_ip, 0)

        # Check if 30 seconds have passed since the last access
        if time.time() - last_access_time < 20:
            return JSONResponse(content={"detail": "Access count exceeded, please try again in 20 seconds"}, status_code=401)
        # Update the last access timestamp
        ip_last_access[client_ip] = time.time()

    # Call the next middleware or route handler
    response = await call_next(request)
    return response
"""
@app.get("/")
async def hello():
    return RedirectResponse("/docs")

@app.get("/image")
async def get_image(request:Request,token:str=None,filename:str=None,prompt:str=None,mode:int=None):
    client_ip = request.client.host

    # Get the last access timestamp for this IP
    last_access_time = ip_last_access.get(client_ip, 0)

    # Check if 30 seconds have passed since the last access
    if time.time() - last_access_time < 20:
        return JSONResponse(content={"detail": "Access count exceeded, please try again in 20 seconds"}, status_code=401)
    # Update the last access timestamp
    ip_last_access[client_ip] = time.time()
    json_load = json.load(open('./token/token.json','r'))
    if (not token in json_load) or (not token):
        print("Token Invalid")
        raise HTTPException(status_code=403, detail="Token invalid")
    if not prompt:
        print("Prompt Invalid")
        raise HTTPException(status_code=403, detail="Prompt Invalid")
    if filename:
        f = filename.replace(".png","").replace(".jpeg","").replace(".jpg","")
    else:
        f = prompt
    
    if mode and mode in [0,1]:
        create_mode = mode
    else:
        create_mode = 0
    print(prompt)
    await create_image(prompt,f,create_mode)
    
    image_path = f"./result/{f}.png"
    # Check if the image file exists
    if os.path.exists(image_path):
        # Get the original image size
        image_size = os.path.getsize(image_path)
        return FileResponse(image_path, media_type="image/png",headers={"Content-Length": str(image_size)})
    else:
        raise HTTPException(404,"failed create image.")