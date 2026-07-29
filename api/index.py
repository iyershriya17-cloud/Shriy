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

# Target active flagship Groq model
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
    """Renders the main Educational Platform interface."""
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
    
    flr_summary = "No major solar flares detected recently."
    cme_summary = "Coronal Mass Ejection activity is currently normal."
    has_active_flare = False
    
    try:
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
        flr_summary = f"Solar flare data unavailable: {str(e)}"

    try:
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
        cme_summary = f"CME data unavailable: {str(e)}"

    return jsonify({
        "summary": f"{flr_summary} {cme_summary}",
        "has_active_flare": has_active_flare,
        "flr_data": flr_summary,
        "cme_data": cme_summary,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Runs spacecraft design and mission evaluation simulation using Groq AI."""
    data = request.get_json() or {}
    
    budget = data.get('budget', 1500)
    propulsion = data.get('propulsion', 40)
    materials = data.get('materials', 30)
    payload = data.get('payload', 30)
    destination = data.get('destination', 'Mars Orbit')
    weather_summary = data.get('weather_summary', 'Normal space weather.')

    if not groq_client:
        return jsonify({
            "launch_status": "GO" if (propulsion >= 30 and materials >= 30 and payload >= 20) else "NO-GO",
            "readiness_rating": min(95, max(20, int(propulsion * 0.8 + materials * 0.9 + payload * 0.7))),
            "allocation_propulsion_m": round(budget * (propulsion / 100.0), 2),
            "allocation_materials_m": round(budget * (materials / 100.0), 2),
            "allocation_payload_m": round(budget * (payload / 100.0), 2),
            "post_mortem_log": "SIMULATION COMPLETED (Offline Engine Fallback): Standard baseline parameters evaluated.",
            "recommended_actions": [
                "Calibrate Groq API Key in environment settings for live AI evaluation.",
                "Review your budget distributions to ensure a balanced spacecraft."
            ]
        })

    system_prompt = (
        "You are AstroForge AI, an expert aerospace engineering professor mentoring undergraduate students. "
        "Review the student's spacecraft budget allocations and current space weather context to evaluate if their mission is viable. "
        "Keep your language educational, clear, and encouraging. Avoid overly cryptic jargon. "
        "You MUST respond ONLY with valid JSON in the exact structure requested:\n"
        "{\n"
        '  "launch_status": "GO" or "NO-GO",\n'
        '  "readiness_rating": integer between 0 and 100,\n'
        '  "allocation_propulsion_m": number,\n'
        '  "allocation_materials_m": number,\n'
        '  "allocation_payload_m": number,\n'
        '  "post_mortem_log": "Detailed educational analysis of why this configuration works or fails",\n'
        '  "recommended_actions": ["action item 1", "action item 2", "action item 3"]\n'
        "}"
    )

    user_message = (
        f"Student Mission Plan:\n"
        f"- Target Destination: {destination}\n"
        f"- Total Budget: ${budget} Million\n"
        f"- Mission Configuration: Engine Budget={propulsion}%, Protection Budget={materials}%, Payload Budget={payload}%\n"
        f"- Current Space Weather Context: {weather_summary}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content
        parsed_result = json.loads(response_content)
        return jsonify(parsed_result)

    except Exception as e:
        return jsonify({
            "launch_status": "ERROR",
            "readiness_rating": 0,
            "allocation_propulsion_m": round(budget * (propulsion / 100.0), 2),
            "allocation_materials_m": round(budget * (materials / 100.0), 2),
            "allocation_payload_m": round(budget * (payload / 100.0), 2),
            "post_mortem_log": f"AI Evaluation Engine Interruption: {str(e)}",
            "recommended_actions": [
                "Verify your configuration parameters and try again."
            ]
        })

@app.route('/api/chat', methods=['POST'])
def chat_copilot():
    """Powers the Engineering Copilot AI Tutor for interactive Q&A."""
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    context_module = data.get('context_module', 'General Aerospace Engineering')

    if not message:
        return jsonify({"response": "Please ask a question about aerospace engineering or space mission architecture."}), 400

    if not groq_client:
        return jsonify({
            "response": f"I am your AI Aerospace Tutor. Currently running in offline mode. Regarding '{message}': In aerospace design, every trade-off balances mass, energy, and structural reliability. Please ensure GROQ_API_KEY is configured for dynamic response generation."
        })

    system_prompt = (
        "You are the AstroForge Engineering Copilot, a friendly and approachable university teaching assistant in aerospace engineering. "
        "Your role is to guide undergraduate students through core concepts like orbital mechanics, propulsion, rocket anatomy, launch dynamics, space weather, and spacecraft design trade-offs. "
        "CRITICAL FORMATTING INSTRUCTIONS - STRICTLY FOLLOW THESE RULES:\n"
        "1. Respond using plain HTML-ready text. Use <br><br> tags to separate short, concise paragraphs.\n"
        "2. Do NOT use any Markdown syntax whatsoever. Absolutely NO asterisks (** or *), NO hash symbols for headers (## or #), NO numbered markdown lists (1., 2., 3.), NO tables, and NO code blocks (```).\n"
        "3. Keep your tone conversational, educational, and easy to read like a friendly university teaching assistant.\n"
        "4. Use simple section titles as plain text followed by <br> without any formatting symbols.\n"
        "5. Include occasional emojis where appropriate to keep it engaging 🚀.\n"
        "6. Use real-world analogies when helpful to explain complex ideas naturally.\n"
        "7. Avoid long information dumps. Keep explanations concise and highlight only the most important ideas.\n"
        "8. When listing items, you must ONLY use simple bullet characters like '•' followed by a space.\n"
        "9. MANDATORY ENDING: You must end every single response with exactly one engaging follow-up question asking if the student would like an example, a diagram, a quick quiz, or a deeper explanation."
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: Module '{context_module}'\nStudent Question: {message}"}
            ],
            temperature=0.5,
            max_tokens=600
        )
        ai_reply = completion.choices[0].message.content
        return jsonify({"response": ai_reply})
    except Exception as e:
        return jsonify({"response": f"AI Tutor temporarily unavailable: {str(e)}"}), 500


ACADEMY_QUESTION_BANK = {
    1: [
        {
            "question": "Why are liquid propellants like liquid oxygen and hydrogen chilled to extreme cryogenic temperatures?",
            "options": [
                "To cool down the rocket's external metal skin during flight",
                "To maximize liquid density, allowing maximum propellant mass to fit inside internal tanks",
                "To reduce pressure inside the engines safely",
                "To prevent fuel from evaporating in microgravity"
            ],
            "correct": 1,
            "reasoning": "Cooling gases until they liquefy drastically increases their density. This allows engineers to pack maximum fuel mass into smaller, lighter tanks."
        },
        {
            "question": "What primary purpose does flooding the launchpad with massive amounts of water serve during liftoff?",
            "options": [
                "To extinguish grass fires around the launch facility",
                "To absorb and dampen acoustic shock waves that could structurally shatter the rocket",
                "To clean carbon soot off the engine bells",
                "To convert exhaust gas into harmless steam"
            ],
            "correct": 1,
            "reasoning": "Rocket engines produce sound energy exceeding 200 decibels. Sound suppression water deluge systems absorb acoustic energy so waves don't bounce off the pad and damage the vehicle."
        }
    ],
    2: [
        {
            "question": "Why do orbital spacecraft need to reach a high horizontal velocity rather than just flying straight up?",
            "options": [
                "To avoid satellite debris in lower atmospheric bands",
                "To enter continuous freefall around Earth where forward velocity matches Earth's curvature",
                "To ensure engines stay cool in the vacuum of space",
                "To direct solar panels toward sunlight continuously"
            ],
            "correct": 1,
            "reasoning": "Orbiting isn't just about reaching space height—it's about moving sideways fast enough (~28,000 km/h) that as gravity pulls the craft down, Earth curves away underneath it at the exact same rate."
        },
        {
            "question": "What distinguishes Low Earth Orbit (LEO) from Geostationary Earth Orbit (GEO)?",
            "options": [
                "LEO is reserved exclusively for military satellites",
                "GEO satellites orbit at ~35,786 km and stay fixed over one geographical spot on Earth",
                "LEO requires double the Delta-v of GEO",
                "GEO is inside Earth's upper atmosphere"
            ],
            "correct": 1,
            "reasoning": "At ~35,786 km altitude, a satellite's orbital period matches Earth's 24-hour rotational period, causing it to appear stationary over a single point on the equator."
        }
    ],
    3: [
        {
            "question": "In a spacecraft mission design team, what is the core responsibility of the Guidance, Navigation, and Control (GNC) Engineer?",
            "options": [
                "Managing financial budgets and vendor contracts",
                "Calculating trajectories, tracking position, and controlling spacecraft attitude/pointing",
                "Designing solar panel silicon arrays",
                "Writing launch marketing documentation"
            ],
            "correct": 1,
            "reasoning": "GNC engineers design the sensors, algorithms, and thruster effectors that determine where the spacecraft is, where it needs to go, and how it points its antennas or thrusters."
        },
        {
            "question": "Why is the Flight Director considered the ultimate authority during live mission launch operations?",
            "options": [
                "They own the financial company executing the launch",
                "They synthesize real-time assessments from all technical console leads to make fast GO/NO-GO decisions",
                "They personally code the flight computer software",
                "They operate the ground telemetry antennas manually"
            ],
            "correct": 1,
            "reasoning": "The Flight Director leads the mission team, listening to specialized console officers (Propulsion, Avionics, Trajectory) to render unified, safety-critical operational calls."
        }
    ],
    4: [
        {
            "question": "What aerodynamic phenomenon is designated by the term 'Max-Q' during atmospheric flight?",
            "options": [
                "The point where engines consume maximum propellant per second",
                "Maximum Dynamic Pressure, representing the peak mechanical stress on the rocket structure",
                "Maximum Quantum radiation exposure in the upper atmosphere",
                "The point where the vehicle breaks the sound barrier"
            ],
            "correct": 1,
            "reasoning": "Max-Q occurs when the combination of atmospheric density and vehicle speed creates the highest physical pressure on the rocket's skin and frame."
        },
        {
            "question": "According to the Tsiolkovsky Rocket Equation, why is multi-stage rocket design necessary for orbital access?",
            "options": [
                "Single-stage rockets are forbidden by international space law",
                "Discarding empty propellant tanks removes useless dead weight, dramatically improving acceleration efficiency",
                "Multiple engines prevent fuel line icing",
                "It allows the rocket to change direction without using RCS thrusters"
            ],
            "correct": 1,
            "reasoning": "Carrying heavy, empty metal tanks all the way to orbit wastes huge amounts of energy. Staging discards spent structure so remaining fuel pushes a much lighter vehicle."
        }
    ],
    5: [
        {
            "question": "Why are Coronal Mass Ejections (CMEs) dangerous to unshielded interplanetary spacecraft?",
            "options": [
                "They increase atmospheric drag at LEO altitudes",
                "They release massive clouds of magnetized high-energy plasma that can disrupt electronics and harm astronauts",
                "They create intense physical wind pressures in deep space vacuum",
                "They freeze the spacecraft's liquid propellants"
            ],
            "correct": 1,
            "reasoning": "CMEs spew billion-ton clouds of energized particles and magnetic fields. Without magnetic fields or physical radiation shielding, spacecraft electronics can fry and humans suffer radiation sickness."
        },
        {
            "question": "How does NASA's DONKI system assist space mission planners?",
            "options": [
                "By tracking asteroid mining rights",
                "By providing real-time telemetry and predictive space weather monitoring for solar flares and CMEs",
                "By calculating launch vehicle manufacturing costs",
                "By scheduling astronaut sleep rotations"
            ],
            "correct": 1,
            "reasoning": "NASA's Space Weather Database Of Notifications, Knowledge, Information (DONKI) tracks space weather events to protect orbiting assets and space crews."
        }
    ],
    6: [
        {
            "question": "In spacecraft design, what is meant by an 'engineering trade-off'?",
            "options": [
                "Exchanging spare parts with international space agencies",
                "Balancing competing constraints like mass, power, and cost—e.g., adding shield mass reduces scientific payload capacity",
                "Selling old rocket designs to private companies",
                "Trading fuel between first and second stages mid-flight"
            ],
            "correct": 1,
            "reasoning": "Every kilogram added to one system (like heavy radiation shielding) must be removed from another (like science instruments or fuel) due to overall launch mass limits."
        },
        {
            "question": "What is 'Delta-v' (Δv) in mission planning?",
            "options": [
                "The change in rocket temperature during re-entry",
                "The total velocity change a spacecraft can achieve with its available onboard propellants",
                "The difference in pressure between engines",
                "The rate of atmospheric density decay"
            ],
            "correct": 1,
            "reasoning": "Delta-v is the scalar measure of impulse needed to perform orbital maneuvers (like changing orbits or burning for Mars). It represents a spacecraft's total 'maneuver budget'."
        }
    ]
}


def get_randomized_questions(module_id):
    """Retrieves and shuffles question options for dynamic student assessment."""
    q_list = ACADEMY_QUESTION_BANK.get(module_id, ACADEMY_QUESTION_BANK[1])
    processed = []
    for q in q_list:
        q_copy = dict(q)
        processed.append(q_copy)
    return processed


@app.route('/api/academy', methods=['GET'])
def get_academy_content():
    """Serves structured educational modules for the Aerospace Rocket Academy."""
    modules = [
        {
            "id": 1,
            "title": "Module 1: Rocket Fundamentals",
            "subtitle": "Anatomy, Propulsion, & Structural Systems",
            "badge": "BASICS",
            "intro": "Every orbital launch vehicle is an extreme engineering balancing act combining high-energy thermodynamics, lightweight structures, and avionics.",
            "lesson": "A rocket's primary job is to deliver a payload into space. To achieve this, it consists of four major sub-systems: Propulsion (engines and cryogenic fuel tanks), Structures (frame and aerodynamic fairings), Guidance & Control (flight computers and grid fins), and Payload (satellites or crew capsules). Liquid-propellant engines combine super-cooled fuels like Liquid Hydrogen (LH2) or Methane (CH4) with Liquid Oxygen (LOX) in a combustion chamber, expelling gas at hypersonic speeds to produce thrust according to Newton's Third Law.",
            "fact": "A Saturn V rocket burned roughly 20 tons of propellant per second at liftoff—93% of its total launch mass was pure fuel!",
            "summary": "Rockets rely on reaction mass expelled through bell nozzles. Structures are kept as thin as soda cans to maximize payload efficiency.",
            "visualType": "anatomy"
        },
        {
            "id": 2,
            "title": "Module 2: Space Missions & Orbits",
            "subtitle": "Orbital Mechanics & Trajectories",
            "badge": "ORBITS",
            "intro": "Reaching space is only half the battle; staying in space requires mastering orbital mechanics.",
            "lesson": "An orbit is a continuous state of freefall. When a rocket accelerates horizontally to ~7.8 km/s (17,500 mph) in Low Earth Orbit (LEO), the Earth curves downward at the exact same rate the spacecraft falls toward it. Higher orbits like Geostationary Earth Orbit (GEO) at 35,786 km require additional energy (Delta-v) but allow satellites to remain fixed above a single terrestrial coordinate. Interplanetary missions to Mars or Lunar transfers require precise Hohmann Transfer trajectories that utilize gravitational wells.",
            "fact": "At LEO speeds, astronauts aboard the International Space Station experience 16 sunrises and sunsets every 24 hours.",
            "summary": "Orbital altitude determines period and speed. Trajectory maneuvers require calculated burns of velocity called Delta-v.",
            "visualType": "orbit"
        },
        {
            "id": 3,
            "title": "Module 3: Space Mission Team Roles",
            "subtitle": "Engineering Disciplines & Operations",
            "badge": "ROLES",
            "intro": "No spacecraft flies alone. Successful spaceflight relies on multi-disciplinary engineering teams working in synchronized harmony.",
            "lesson": "In aerospace engineering, specialized teams oversee distinct subsystems: Flight Directors hold ultimate decision authority; Mission Planners optimize trajectory budgets; Propulsion Engineers manage turbopumps and chamber pressures; Guidance, Navigation & Control (GNC) Engineers write attitude-determination algorithms; Avionics Engineers build radiation-tolerant flight computers; and Payload Engineers ensure scientific instruments survive launch vibration environments.",
            "fact": "During Apollo 11's lunar landing, 24-year-old Guidance Officer Steve Bales saved the mission by correctly identifying the 1202 computer overload code in seconds.",
            "summary": "Space missions demand seamless cross-talk between Propulsion, GNC, Thermal, Avionics, and Flight Operations disciplines.",
            "visualType": "team"
        },
        {
            "id": 4,
            "title": "Module 4: Rocket Launch Process",
            "subtitle": "Ignition, Max-Q, & Staging Mechanics",
            "badge": "LAUNCH",
            "intro": "From pad water deluge to engine cutoff, the ascent profile is a tightly orchestrated sequence of physical milestones.",
            "lesson": "The launch countdown leads to ignition, where liquid turbopumps spin at up to 30,000 RPM. Immediately after liftoff, the vehicle executes a 'Gravity Turn'—gently tilting sideways to let gravity naturally turn its velocity vector horizontal. As atmospheric density drops and velocity rises, the rocket passes through 'Max-Q' (Maximum Dynamic Pressure), where throttle is reduced to prevent structural fatigue. Once first stage fuel empties, staging occurs: spent boosters drop away and second-stage vacuum engines ignite (SECO).",
            "fact": "During Max-Q, dynamic atmospheric stress on a rocket can exceed 30 kilopascals—equivalent to hundreds of tons pushing against the vehicle nose.",
            "summary": "Gravity turns minimize aerodynamic drag losses. Staging sheds dead structure mass to unlock orbital acceleration.",
            "visualType": "launch"
        },
        {
            "id": 5,
            "title": "Module 5: Space Weather Telemetry",
            "subtitle": "Solar Flares, CMEs, & Radiation Shielding",
            "badge": "WEATHER",
            "intro": "Space is not a calm vacuum—it is constantly blasted by high-energy solar radiation and magnetic storms.",
            "lesson": "Solar Flares release massive pulses of X-rays and ultraviolet radiation at light speed, while Coronal Mass Ejections (CMEs) spew billions of tons of magnetized solar plasma across interplanetary space. When solar particles hit spacecraft, they can cause Single Event Upsets (SEUs) in microchips or accumulate electrical charge that short-circuits power grids. Aerospace engineers use NASA DONKI telemetry to monitor solar activity and equip spacecraft with aluminum, polyethylene, or magnetic shielding.",
            "fact": "The Carrington Event of 1859 was a CME so intense it caused aurora borealis as far south as Hawaii and sparked telegraph wires worldwide.",
            "summary": "Space weather poses direct hazards to electronics and astronauts. NASA DONKI provides crucial early warnings.",
            "visualType": "weather"
        },
        {
            "id": 6,
            "title": "Module 6: Mission Planning & Trade-offs",
            "subtitle": "Budget Distributions & Engineering Balances",
            "badge": "CAPSTONE",
            "intro": "Every aerospace mission is governed by hard physical constraints: Mass vs. Power vs. Financial Budget.",
            "lesson": "Engineering trade-offs are central to mission design. Allocating too much budget to heavy Engine Propulsion limits the weight available for Radiation Shielding or Science Payloads. Conversely, under-investing in engine performance risks leaving the craft stranded short of its target orbit. Before executing a mission simulation, engineers create trade-off matrices to optimize performance across target destinations like Mars, the Moon, or Europa.",
            "fact": "NASA's James Webb Space Telescope took over 20 years of trade-off optimization to balance its 6.5-meter golden mirror against Ariane 5 rocket payload fairing size constraints.",
            "summary": "Successful mission architecture balances engine performance, structural protection, and payload capabilities within fixed financial limits.",
            "visualType": "tradeoff"
        }
    ]

    for mod in modules:
        mod["questions"] = get_randomized_questions(mod["id"])

    return jsonify({"modules": modules, "status": "OK"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
