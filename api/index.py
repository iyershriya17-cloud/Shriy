import os
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from local .env
load_dotenv()

app = Flask(__name__, template_folder='../templates')

# Target active flagship Groq model for ultra-low latency calculations
GROQ_MODEL = "openai/gpt-oss-120b"

# Initialize AI compute engine securely
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# NASA Open API Ingestion Parameters
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
NASA_DONKI_FLR_URL = "https://api.nasa.gov/DONKI/FLR"
NASA_DONKI_CME_URL = "https://api.nasa.gov/DONKI/CME"
NASA_IMAGE_API_URL = "https://images-api.nasa.gov/search"

# Supabase & Google Configuration Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

@app.route('/')
def home():
    """Renders the main Mission Control Room terminal HUD interface."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """System diagnostic monitoring hook for platform operations."""
    return jsonify({
        "status": "OPERATIONAL",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "groq_configured": groq_client is not None,
        "nasa_api_active": bool(NASA_API_KEY),
        "auth_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY)
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Exposes public client configurations for Supabase and Google Auth."""
    return jsonify({
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "google_client_id": GOOGLE_CLIENT_ID
    })

@app.route('/api/weather', methods=['GET'])
def get_space_weather():
    """Fetches real-time space weather telemetry from NASA DONKI APIs."""
    end_date = datetime.utcnow().strftime('%Y-%m-%d')
    start_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    flr_summary = "No major solar flares detected in current telemetry window."
    cme_summary = "Coronal Mass Ejection activity within nominal background levels."
    has_active_flare = False
    
    try:
        # Fetch Solar Flare data
        flr_response = requests.get(
            NASA_DONKI_FLR_URL,
            params={"startDate": start_date, "endDate": end_date, "api_key": NASA_API_KEY},
            timeout=5
        )
        if flr_response.status_code == 200 and flr_response.json():
            flares = flr_response.json()
            if len(flares) > 0:
                latest_flr = flares[-1]
                flr_class = latest_flr.get('classType', 'Unknown')
                begin_time = latest_flr.get('beginTime', 'N/A')
                flr_summary = f"Active Class {flr_class} Solar Flare detected at {begin_time}."
                has_active_flare = True
    except Exception as e:
        flr_summary = f"FLR Telemetry stream degraded: {str(e)}"

    try:
        # Fetch Coronal Mass Ejection data
        cme_response = requests.get(
            NASA_DONKI_CME_URL,
            params={"startDate": start_date, "endDate": end_date, "api_key": NASA_API_KEY},
            timeout=5
        )
        if cme_response.status_code == 200 and cme_response.json():
            cmes = cme_response.json()
            if len(cmes) > 0:
                latest_cme = cmes[-1]
                time21 = latest_cme.get('startTime', 'N/A')
                cme_summary = f"Coronal Mass Ejection registered at {time21}."
    except Exception as e:
        cme_summary = f"CME Telemetry stream degraded: {str(e)}"

    return jsonify({
        "summary": f"{flr_summary} {cme_summary}",
        "has_active_flare": has_active_flare,
        "flr_data": flr_summary,
        "cme_data": cme_summary,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Runs spacecraft design and mission telemetry simulation using Groq AI LLM reasoning."""
    data = request.get_json() or {}
    
    budget = data.get('budget', 1500)
    propulsion = data.get('propulsion', 40)
    materials = data.get('materials', 30)
    payload = data.get('payload', 30)
    destination = data.get('destination', 'Mars Orbit')
    weather_summary = data.get('weather_summary', 'Nominal solar magnetic field.')

    if not groq_client:
        # Fallback simulation response if Groq API Key is not set
        return jsonify({
            "launch_status": "GO" if (propulsion >= 30 and materials >= 30 and payload >= 20) else "NO-GO",
            "readiness_rating": min(95, max(20, int(propulsion * 0.8 + materials * 0.9 + payload * 0.7))),
            "allocation_propulsion_m": round(budget * (propulsion / 100.0), 2),
            "allocation_materials_m": round(budget * (materials / 100.0), 2),
            "allocation_payload_m": round(budget * (payload / 100.0), 2),
            "post_mortem_log": "SIMULATION COMPLETED (Offline Engine Fallback): Standard baseline parameters evaluated.",
            "recommended_actions": [
                "Calibrate Groq API Key in environment settings for live deep AI telemetry analytics.",
                "Ensure shield thickness meets radiation tolerance for target trajectory."
            ]
        })

    system_prompt = (
        "You are AstroForge AI, an expert aerospace systems architect and mission director for interplanetary spacecraft. "
        "Analyze the provided mission architecture, budget allocations, and space weather context to evaluate launch feasibility. "
        "You MUST respond ONLY with valid JSON in the exact structure requested:\n"
        "{\n"
        '  "launch_status": "GO" or "NO-GO",\n'
        '  "readiness_rating": integer between 0 and 100,\n'
        '  "allocation_propulsion_m": number,\n'
        '  "allocation_materials_m": number,\n'
        '  "allocation_payload_m": number,\n'
        '  "post_mortem_log": "Detailed architectural analysis and flight risk breakdown string",\n'
        '  "recommended_actions": ["action item 1", "action item 2", "action item 3"]\n'
        "}"
    )

    user_message = (
        f"Mission Telemetry Request:\n"
        f"- Destination Target: {destination}\n"
        f"- Total Allocated Budget: ${budget} Million\n"
        f"- Parameter Priorities: Propulsion Systems={propulsion}%, Raw Materials/Shielding={materials}%, Avionics/Payload={payload}%\n"
        f"- Current Environmental Tracking Radar: {weather_summary}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.25,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content
        parsed_result = json.loads(response_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({
            "launch_status": "NO-GO",
            "readiness_rating": 5,
            "allocation_propulsion_m": round(budget * (propulsion / 100.0), 2),
            "allocation_materials_m": round(budget * (materials / 100.0), 2),
            "allocation_payload_m": round(budget * (payload / 100.0), 2),
            "post_mortem_log": f"Critical Telemetry Engine Interruption Exception: {str(e)}",
            "recommended_actions": [
                "Verify system parameters or check connection configurations.",
                "Ensure API environment variables are properly exported in deployment configuration."
            ]
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)