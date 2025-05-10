import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "text": "Breaking: President signs new economic relief bill into law."
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
