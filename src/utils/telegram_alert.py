import requests

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "5283049125"

def send_alert(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass  
