import os
import json
import random
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

# Master Quiz Question Bank (pool per phase id). A random sample of 2-4 is
# served per request so repeated visits to a phase feel fresh. Users are
# always allowed to advance regardless of whether answers are right or wrong;
# the "correct" and "reasoning" fields let the frontend reveal the answer and
# explain why immediately after the user selects an option.
ACADEMY_QUESTION_BANK = {
    1: [
        {
            "question": "Why are liquid propellant tanks (LOX/LH2) chilled to cryogenic temperatures prior to launch?",
            "options": [
                "To cool down the rocket metal skin",
                "To maximize liquid propellant density so more fuel mass fits in the tanks",
                "To lower pressure inside the propellant lines",
                "To prevent fuel from evaporating in microgravity"
            ],
            "correct": 1,
            "reasoning": "Cryogenic cooling dramatically increases propellant density. Liquid oxygen and hydrogen take up significantly less volume when liquid, allowing rockets to pack maximum fuel mass into lightweight structural tanks."
        },
        {
            "question": "What is the primary operational purpose of the launchpad Sound Suppression Water Deluge system?",
            "options": [
                "To extinguish pad grass fires",
                "To dampen destructive acoustic shock waves (>200 dB) that could tear apart the rocket hull",
                "To wash off rocket nozzle soot",
                "To cool down the exhaust gas into water vapor"
            ],
            "correct": 1,
            "reasoning": "At liftoff, rocket engines generate sound energy over 200 dB. Dumping tens of thousands of gallons of water per second dampens sound waves that would otherwise bounce off concrete and destroy delicate onboard avionics."
        },
        {
            "question": "What occurs during the autonomous Terminal Countdown sequence triggered at T-Minus 10 minutes?",
            "options": [
                "Flight control transitions to onboard guidance computers and internal battery power",
                "Astronauts execute manual engine throttle adjustments",
                "Satellite payloads are powered down",
                "Ground crews manually disconnect fueling valves"
            ],
            "correct": 0,
            "reasoning": "In the final minutes of countdown, flight control switches automatically from ground servers to the vehicle's autonomous flight computers. Ground power umbilicals detach as onboard batteries take over."
        },
        {
            "question": "Why are spark igniters fired below engine nozzles seconds BEFORE engine clamp release?",
            "options": [
                "To test battery voltage",
                "To burn off residual unburned hydrogen gas and prevent explosive hard starts",
                "To heat up the rocket payload",
                "To signal ground personnel"
            ],
            "correct": 1,
            "reasoning": "Pre-ignition sparkers burn off any lingering hydrogen/methane gas around the engine bell, preventing dangerous explosive shock detonations ('hard starts') when main fuel valves open."
        }
    ],
    2: [
        {
            "question": "What does 'Max-Q' signify during a rocket's atmospheric ascent trajectory?",
            "options": [
                "Maximum rate of fuel consumption",
                "The exact point of Maximum Dynamic Pressure where aerodynamic drag forces peak",
                "Maximum orbital altitude achieved",
                "Maximum radiation exposure in the magnetosphere"
            ],
            "correct": 1,
            "reasoning": "Max-Q occurs when high vehicle speed meets thick atmosphere. The combination of accelerating velocity and air density creates the absolute peak mechanical stress load on the rocket's hull."
        },
        {
            "question": "Why do rocket main engines throttle down slightly as the vehicle approaches Max-Q?",
            "options": [
                "To conserve fuel reserves for orbit",
                "To reduce dynamic aerodynamic loads and avoid structural hull failure",
                "To allow turbopumps to cool off",
                "To improve radio telemetry signals"
            ],
            "correct": 1,
            "reasoning": "Throttling engines back slightly reduces acceleration forces precisely when atmospheric drag is at its maximum intensity. Once past Max-Q into thinner air, full thrust is restored."
        },
        {
            "question": "What creates the dramatic cone-shaped vapor cloud (Prandtl-Glauert effect) near Max-Q?",
            "options": [
                "Engine exhaust smoke leaking upward",
                "Local pressure drops at transonic speeds causing atmospheric water vapor to instantly condense",
                "Cryogenic fuel tank venting",
                "Thermal protection tiles scorching"
            ],
            "correct": 1,
            "reasoning": "As the rocket approaches Mach 1 near Max-Q, rapid pressure drops around the expanding fairing cause moisture in ambient air to condense into a visible cone shockwave cloud."
        },
        {
            "question": "Why does a rocket perform a 'Gravity Turn' rather than flying straight up into space?",
            "options": [
                "To avoid active satellite constellations",
                "To convert vertical kinetic energy into horizontal velocity required for orbit while minimizing steering losses",
                "To keep the rocket over water",
                "To point solar panels toward the sun"
            ],
            "correct": 1,
            "reasoning": "Achieving orbit requires reaching ~28,000 km/h of horizontal speed. A gravity turn uses Earth's gravitational pull to smoothly curve the vehicle trajectory sideways without wasting steering fuel."
        }
    ],
    3: [
        {
            "question": "What event occurs at MECO (Main Engine Cut-Off)?",
            "options": [
                "An emergency abort signal",
                "Shutdown of first-stage engines prior to pneumatic staging separation",
                "Final spacecraft burn in orbit",
                "Payload deployment trigger"
            ],
            "correct": 1,
            "reasoning": "MECO halts first-stage main engine thrust after burning the majority of propellant. Stopping thrust ensures a clean separation of the heavy booster from the upper stage."
        },
        {
            "question": "According to Tsiolkovsky's Rocket Equation, why drop the spent first stage booster?",
            "options": [
                "It is required by international space law",
                "Shedding dead structural mass improves the remaining stage's mass ratio and achievable velocity",
                "It reduces radio interference",
                "It keeps the booster from overheating"
            ],
            "correct": 1,
            "reasoning": "The rocket equation shows final velocity depends heavily on the ratio of starting mass to remaining mass. Dropping an empty, heavy booster lets the upper stage accelerate far more efficiently with its own fuel."
        },
        {
            "question": "Why do ullage thrusters fire briefly before second-stage engine ignition?",
            "options": [
                "To slow the vehicle down for staging",
                "To settle liquid propellant toward the tank outlets in zero-gravity, preventing pump cavitation",
                "To realign the guidance computer",
                "To vent excess oxygen"
            ],
            "correct": 1,
            "reasoning": "In zero-g, propellant floats freely inside the tank. Small ullage thrusters provide gentle acceleration that pushes fuel toward the outlet, ensuring the main engine turbopumps ingest liquid instead of gas bubbles."
        }
    ],
    4: [
        {
            "question": "Why is the payload fairing jettisoned once the vehicle crosses the Karman Line?",
            "options": [
                "To reduce radio signal interference",
                "Dense atmospheric drag has disappeared, so the protective shielding becomes unnecessary dead weight",
                "To expose solar panels early",
                "To trigger the next stage's ignition sequence"
            ],
            "correct": 1,
            "reasoning": "Below 100 km, the fairing protects the payload from aerodynamic heating and pressure. Once in near-vacuum, that protection is no longer needed, so jettisoning it maximizes payload mass carried to orbit."
        },
        {
            "question": "What material is the payload fairing typically constructed from, and why?",
            "options": [
                "Solid steel, for maximum durability",
                "Lightweight carbon-composite honeycomb panels, balancing strength with minimal added mass",
                "Reinforced glass, for visibility",
                "Pure aluminum foil, for radio transparency"
            ],
            "correct": 1,
            "reasoning": "Carbon-composite honeycomb structures provide high structural strength while adding minimal mass, which is critical since every extra kilogram reduces how much payload can reach orbit."
        }
    ],
    5: [
        {
            "question": "What is meant by 'orbital velocity' at roughly 28,000 km/h?",
            "options": [
                "The maximum speed a rocket engine can produce",
                "The sideways speed at which a falling object's curved path matches the curvature of the Earth, resulting in continuous freefall around it",
                "The re-entry speed for returning spacecraft",
                "The speed at which fuel tanks depressurize"
            ],
            "correct": 1,
            "reasoning": "Orbit isn't about escaping gravity; it is about moving sideways fast enough that as the spacecraft falls toward Earth, the ground curves away at the same rate, producing a continuous freefall path around the planet."
        },
        {
            "question": "What happens at SECO (Second Engine Cut-Off)?",
            "options": [
                "The rocket begins re-entry",
                "The second stage shuts down thrust after reaching target orbital velocity, allowing payload separation",
                "The mission is aborted",
                "The first stage re-ignites for landing"
            ],
            "correct": 1,
            "reasoning": "SECO marks the moment the upper stage has delivered the spacecraft to its target orbital speed and altitude. Engines cut off and the payload can then safely separate and deploy."
        },
        {
            "question": "Why do spent upper stages perform a 'passivation' and de-orbit burn after payload release?",
            "options": [
                "To recover fuel for reuse",
                "To vent remaining energy and safely de-orbit, reducing the risk of orbital debris and explosions",
                "To boost the payload further",
                "To recharge onboard batteries"
            ],
            "correct": 1,
            "reasoning": "Leftover pressurized propellant or battery energy in a discarded stage can rupture and create debris. Passivation vents these hazards and a de-orbit burn ensures the stage safely re-enters rather than lingering as space junk."
        }
    ]
}


def get_randomized_questions(phase_id):
    """Selects a random 2-4 question subset for a given phase from the bank."""
    pool = ACADEMY_QUESTION_BANK.get(phase_id, [])
    count = min(len(pool), random.randint(2, 4))
    return random.sample(pool, count) if pool else []


@app.route('/api/academy', methods=['GET'])
def get_academy_phases():
    """Provides structured rocket launch educational phase data and interactive diagnostic checks."""
    phases = [
        {
            "id": 1,
            "title": "Phase 1: Pre-Launch Prep & Countdown",
            "subtitle": "Cryogenic Propellant Loading & Terminal Ignition Sequence",
            "badge": "T-MINUS COUNTDOWN",
            "altitude": "0 km",
            "velocity": "0 km/h",
            "dynamic_pressure": "0 kPa",
            "summary": "Pre-launch operations involve chilling rocket tanks with liquid oxygen and liquid hydrogen/RP-1 kerosene, performing flight control computer handoffs at T-10 minutes, and triggering spark igniters to burn off excess unburned gases prior to hold-down clamp release.",
            "key_concepts": [
                "Cryogenic density management for maximum propellant mass",
                "Acoustic water deluge sound suppression (over 200 dB mitigation)",
                "Autonomous onboard flight computer handoff and battery power transition"
            ]
        },
        {
            "id": 2,
            "title": "Phase 2: Liftoff & Max-Q Aerodynamics",
            "subtitle": "Supersonic Acceleration & Peak Atmospheric Stress",
            "badge": "MAX-Q ASCENT",
            "altitude": "12 - 15 km",
            "velocity": "1,600 - 2,200 km/h",
            "dynamic_pressure": "35 - 45 kPa",
            "summary": "At liftoff, the rocket accelerates vertically before initiating a gravity turn trajectory. Max-Q (Maximum Dynamic Pressure) represents the exact flight moment where aerodynamic drag stress on the vehicle hull reaches its absolute peak. Engines throttle back slightly to ensure structural integrity.",
            "key_concepts": [
                "Gravity turn trajectory for optimal orbital kinetic energy conversion",
                "Engine throttling to protect hull against aerodynamic overstress",
                "Prandtl-Glauert shock wave condensation (vapor cone effect) at supersonic speeds"
            ]
        },
        {
            "id": 3,
            "title": "Phase 3: Main Engine Cut-Off (MECO) & Staging",
            "subtitle": "Booster Separation & Second-Stage Vacuum Ignition",
            "badge": "STAGE SEPARATION",
            "altitude": "65 - 80 km",
            "velocity": "7,500 - 9,000 km/h",
            "dynamic_pressure": "< 1 kPa",
            "summary": "When the first stage exhausts its main propellant supply, MECO (Main Engine Cut-Off) is triggered. Pneumatic or pyrotechnic pushers cleanly detach the spent booster. Small ullage thrusters settle propellant in zero-g before the upper-stage vacuum engine ignites.",
            "key_concepts": [
                "Mass ratio optimization via Tsiolkovsky's rocket equation",
                "Vacuum-expanded engine nozzle bells for maximum thrust efficiency (Specific Impulse)",
                "Ullage thruster acceleration to prevent zero-g fuel pump cavitation"
            ]
        },
        {
            "id": 4,
            "title": "Phase 4: Karman Line Fairing Separation",
            "subtitle": "Atmospheric Boundary Crossing & Spacecraft Exposure",
            "badge": "KARMAN LINE CROSSING",
            "altitude": "100 - 120 km",
            "velocity": "12,000 - 16,000 km/h",
            "dynamic_pressure": "~ 0 kPa",
            "summary": "As the vehicle crosses the Karman Line (100 km altitude) into space vacuum, dense air friction disappears. The protective carbon-composite payload fairing halves split apart and jettison, discarding dead weight to maximize orbital payload capacity.",
            "key_concepts": [
                "Aerodynamic and thermal shielding during atmospheric ascent",
                "Pneumatic fairing detachment in near-vacuum space environments",
                "Lightweight carbon-composite honeycomb structural design"
            ]
        },
        {
            "id": 5,
            "title": "Phase 5: Orbital Insertion & Payload Deployment (SECO)",
            "subtitle": "Second Engine Cut-Off & Spacecraft Mission Control Lock",
            "badge": "ORBITAL INSERTION",
            "altitude": "200 - 500 km",
            "velocity": "27,500 - 28,200 km/h",
            "dynamic_pressure": "0 kPa",
            "summary": "The second stage accelerates the vehicle to orbital speed (~28,000 km/h, where sideways velocity matches Earth's curvature). After SECO (Second Engine Cut-Off), the satellite detaches, deploys solar arrays, locks ground tracking communications, and the booster performs a safe de-orbit burn.",
            "key_concepts": [
                "Orbital velocity physics (~7.8 km/s continuous freefall trajectory)",
                "Autonomous solar panel deployment and attitude control stabilization",
                "Passivation and upper-stage de-orbit burns to prevent orbital space debris"
            ]
        }
    ]

    # Attach a fresh random sample of 2-4 quiz questions to each phase.
    # Users can always proceed to the next phase regardless of whether they
    # answer correctly -- "correct" and "reasoning" are included so the
    # frontend can reveal the right answer and explain it immediately.
    for phase in phases:
        phase["questions"] = get_randomized_questions(phase["id"])

    return jsonify({"phases": phases, "status": "OK"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)