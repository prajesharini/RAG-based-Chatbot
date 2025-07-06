import requests

url = "https://www.pwc.com/m1/en/publications/documents/2024/agentic-ai-the-new-frontier-in-genai-an-executive-playbook.pdf"

try:
    response = requests.get(url, timeout=10)
    print(f"✅ PDF downloaded! Size: {len(response.content)} bytes")
except Exception as e:
    print(f"❌ Failed to download PDF: {e}")

