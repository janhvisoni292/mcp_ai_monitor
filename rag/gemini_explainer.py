
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


def explain_report(report_json: dict) -> str:
    """Sends the audit report to Gemini and returns a plain-English explanation."""

    if not API_KEY:
        return "Gemini API Key not found. Add GOOGLE_API_KEY to your .env file."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = f"""
        You are a helpful AI Model Auditor.

        Given the following JSON report about an AI system's health,
        write a SHORT (3-4 sentence) plain-English explanation that:

        1. States the current status (PASS/WARNING/CRITICAL)
        2. Explains WHY it is in that state
        3. Suggests what a non-technical manager should do

        REPORT:
        {report_json}
        """

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"Gemini Error: {str(e)}"
