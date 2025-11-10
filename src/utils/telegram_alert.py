import requests

BOT_TOKEN = "8415463781:AAFK5UkXqcr8K5lWHwYLzQuQ_WqTQePDCMg"
CHAT_ID = "5283049125"

def send_alert(message: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass  
