
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load env from root or relative path
load_dotenv()

# Try to get API key from environment
# User must create a .env file with GOOGLE_API_KEY=...
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

def explain_report(report_json: dict) -> str:
    """
    Uses Gemini Free API to generate a plain-English explanation of the audit report.
    """
    if not API_KEY:
        return "⚠️ Gemini API Key not found. Please add GOOGLE_API_KEY to your .env file to see AI explanations."

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        You are a helpful AI Model Auditor.
        I will give you a JSON report about an AI system's health.
        
        Your task:
        1. Summarize the status (PASS/WARNING/CRITICAL).
        2. Explain WHY it is in that state in simple terms.
        3. If there is drift or bias, explain what that means for a non-technical user.
        4. Keep it short (max 3-4 sentences).

        REPORT:
        {report_json}
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"
