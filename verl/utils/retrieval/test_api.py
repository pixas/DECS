import requests

# URL for your local FastAPI server
url = "http://10.140.0.170:10044/retrieve"

# Example payload
payload = {
    "queries": ["Slow reacting substance of anaphylaxis (SRS-A) is a mix of compounds produced during allergic reactions and asthma."]

}

# Send POST request

response = requests.post(url, json=payload)

# Raise an exception if the request failed
response.raise_for_status()

# Get the JSON response
retrieved_data = response.json()

print("Response from server:")
print(retrieved_data)

# 启用多个线程同时访问api，测试api的并发能力
import threading
import time
from queue import Queue

# Create a queue to store results
results_queue = Queue()

def make_request():
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        # Store thread name and response data in queue
        # only save 
        result = response.json()
        new_result = result.copy()
        if result['embeds'] is None:
            raise ValueError("503HOT！LLM服务器推理繁忙，请稍后再试")
        new_result['embeds'] = (len(result['embeds']), len(result['embeds'][0]))
        results_queue.put({
            'thread': threading.current_thread().name,
            'data': new_result
        })
    except Exception as e:
        results_queue.put({
            'thread': threading.current_thread().name,
            'error': str(e)
        })

# Number of concurrent requests to make
num_threads = 10

# Create and start threads
threads = []
start_time = time.time()

for i in range(num_threads):
    thread = threading.Thread(target=make_request)
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

end_time = time.time()

# Print results from all threads
print("\nResults from each thread:")
while not results_queue.empty():
    result = results_queue.get()
    print(f"\nThread {result['thread']}:")
    if 'data' in result:
        print(result['data'])
    else:
        print(f"Error: {result['error']}")

print(f"\nCompleted {num_threads} concurrent requests in {end_time - start_time:.2f} seconds")