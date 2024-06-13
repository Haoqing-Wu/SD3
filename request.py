import requests
import json
import base64


url = 'http://127.0.0.1:5001/predictions'

# Prepare the payload
payload = json.dumps({
  "input": {
    "prompt": "A cat holding a sign that says hello world",
    "negative_prompt": "",
    "num_inference_steps": 28,
    "height": 1024,
    "width": 1024,
    "guidance_scale": 7.0
  }
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
result = response.json()
content = result.get('output')
header, content = content.split("base64,", 1)
content = base64.b64decode(content)
with open("./output.jpg", "wb") as f:
    f.write(content)