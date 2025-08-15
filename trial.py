import requests

url = "http://127.0.0.1:8000/query"
data = {"question": "What is the syllabus for 2nd year ECE?"}

response = requests.post(url, json=data)
print(response.json())  
