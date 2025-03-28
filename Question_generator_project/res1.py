import requests

url = "http://127.0.0.1:8000/generate-questions"
file_link = "C:/Users/palnati.rahulreddy/Downloads/bin1/output.pdf"
topic = "Hobbits"

data = {'file_link': file_link,'topic': topic}

response = requests.post(url, data=data)

print(response)