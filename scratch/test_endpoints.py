import requests

try:
    print("Testing /api/trajectory...")
    res = requests.get("http://127.0.0.1:8000/api/trajectory?date=2026-07-17")
    print(res.status_code)
    data = res.json()
    print(f"Total points: {len(data)}")
    if data:
        print(data[:3])
except Exception as e:
    print(f"Error trajectory: {e}")

try:
    print("\nTesting /api/transport/stats...")
    res = requests.get("http://127.0.0.1:8000/api/transport/stats?date=2026-07-17")
    print(res.status_code)
    print(res.json())
except Exception as e:
    print(f"Error transport/stats: {e}")
