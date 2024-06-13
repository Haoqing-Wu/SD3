from typing import Optional, Union, List
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from fastapi import File, UploadFile,Response,status
from fastapi import FastAPI, File, UploadFile, APIRouter, Depends, Request, status
from typing_extensions import Annotated
import torch,json
import urllib.request
import time
import base64

from diffusers import StableDiffusion3Pipeline
from PIL import Image
from fastapi.openapi.utils import get_openapi

app = FastAPI(title="SD3",version='0.0.1',docs_url=None)


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  

class Parameters(BaseModel):
    prompt: str = Field(title="Prompt", description="The input prompt for the model. (Default: '')", default="")
    negative_prompt: Optional[str] = Field(title="Negative Prompt", description="The negative input prompt for the model. (Default: '')", default="")
    num_inference_steps: Optional[int] = Field(title="Num Inference Steps", description="The number of inference steps to run. (Default: 28)", default=28)
    height: Optional[int] = Field(title="Height", description="The height of the model input. (Default: 1024)", default=1024)
    width: Optional[int] = Field(title="Width", description="The width of the model input. (Default: 1024)", default=1024)
    guidance_scale: Optional[float] = Field(title="Guidance Scale", description="The guidance scale for the model. (Default: 7.0)", default=7.0)
    

class Result(BaseModel):
    output: str=Field(title="output",  description="output of SD3")
    
class InputData(BaseModel):
    input: Parameters=Field(title="Input")

access_token = "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"
pipe = StableDiffusion3Pipeline.from_pretrained("/sd3/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def image_to_data_uri(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        data = f.read()
    return prefix + base64.b64encode(data).decode('utf-8')


@app.get("/",status_code=status.HTTP_200_OK)
def Root():
    return Response(content='ok')

@app.get("/health-check")
def Healthcheck():
    return Response(content='ok')

@app.post("/predictions",response_model=Result)
def Predict(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. read inputs ...')
    
        
    print('2. Inferring ... ')
    t1 = time.time()
    image = pipe(
        inputdata.input.prompt,
        negative_prompt=inputdata.input.negative_prompt,
        num_inference_steps=inputdata.input.num_inference_steps,
        height=inputdata.input.height,
        width=inputdata.input.width,
        guidance_scale=inputdata.input.guidance_scale,
    ).images[0]
    image.save('output.png')
    output = image_to_data_uri('output.png')
    print('3. Inference time cost:', time.time() - t1)
    
    ret = {
    "output": output
    }
    return ret
    
    
@app.get("/docs", include_in_schema=False)
def get_docs():
    return get_openapi(title="qwen",version='0.0.1', routes=app.routes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=5001)