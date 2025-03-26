import requests

"""
触媒反応を適用
- 反応時間を短縮し、反応収率を向上させることが
"""
url = "http://localhost:8000/generate"
payload = {"text": "触媒反応を適用"}
response = requests.post(url, json=payload)

generated_text = response.json()["generated_text"]

print(generated_text)