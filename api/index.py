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

# Conversation memory tuning: how many prior exchanges (user+assistant pairs)
# we retain and forward to the model on every AI Tutor request.
MAX_HISTORY_MESSAGES = 30  # ~15 exchanges, safely inside the model's context window

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
        "The student has just completed Rocket Academy II (Mission Planning Academy), so you can reference concepts like "
        "engineering trade-offs, Delta-v budgets, mass margins, reliability, and safety margins directly, but still explain "
        "your reasoning clearly. Keep your language educational, clear, and encouraging. Avoid overly cryptic jargon. "
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


def _sanitize_history(raw_history):
    """
    Validates and trims client-supplied conversation history so the AI Tutor
    remembers the ongoing conversation instead of restarting on every message.
    Only well-formed {role, content} pairs with role in (user, assistant) survive.
    """
    if not isinstance(raw_history, list):
        return []

    cleaned = []
    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        role = entry.get('role')
        content = entry.get('content')
        if role not in ('user', 'assistant'):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        cleaned.append({"role": role, "content": content.strip()[:4000]})

    # Keep only the most recent exchanges so we stay comfortably inside the
    # model's context window while still preserving continuity.
    if len(cleaned) > MAX_HISTORY_MESSAGES:
        cleaned = cleaned[-MAX_HISTORY_MESSAGES:]

    return cleaned


@app.route('/api/chat', methods=['POST'])
def chat_copilot():
    """Powers the Engineering Copilot AI Tutor for interactive Q&A with persistent memory."""
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    context_module = data.get('context_module', 'General Aerospace Engineering')
    history = _sanitize_history(data.get('history', []))

    if not message:
        return jsonify({"response": "Please ask a question about aerospace engineering or space mission architecture."}), 400

    if not groq_client:
        return jsonify({
            "response": f"I am your AI Aerospace Tutor. Currently running in offline mode. Regarding '{message}': In aerospace design, every trade-off balances mass, energy, and structural reliability. Please ensure GROQ_API_KEY is configured for dynamic response generation."
        })

    system_prompt = (
        "You are the AstroForge Engineering Copilot, a friendly and approachable university teaching assistant in aerospace engineering. "
        "Your role is to guide undergraduate students through core concepts like orbital mechanics, propulsion, rocket anatomy, launch dynamics, space weather, and spacecraft design trade-offs. "
        "You are having an ONGOING conversation with this student. Conversation history is provided to you as prior turns — "
        "use it to understand short follow-ups like 'yes', 'no', 'explain that', 'tell me more', 'why?', 'give me an example', "
        "or 'can you simplify that?' in the context of whatever was just discussed. Never ask the student to repeat context "
        "they already gave you earlier in the conversation.\n"
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

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context: Module '{context_module}'\nStudent Question: {message}"})

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=600
        )
        ai_reply = completion.choices[0].message.content
        return jsonify({"response": ai_reply})
    except Exception as e:
        return jsonify({"response": f"AI Tutor temporarily unavailable: {str(e)}"}), 500


# ====================================================================
# ROCKET ACADEMY I — AEROSPACE FOUNDATIONS
# Ten modules taking a student from zero prior knowledge through the
# core physics and engineering vocabulary of aerospace engineering.
# ====================================================================

ACADEMY_1_MODULES = [
    {
        "id": 1,
        "title": "Module 1: Introduction to Aerospace Engineering",
        "subtitle": "What the Field Is, and Why It Exists",
        "badge": "FOUNDATIONS",
        "intro": "Aerospace engineering is the discipline of designing vehicles that operate inside the atmosphere (aeronautics) and beyond it (astronautics) — from gliders to interplanetary probes.",
        "lesson": "<p>Let's start from zero. Aerospace engineering is the branch of engineering concerned with the design, construction, and operation of vehicles that fly — whether through air (aeronautical engineering) or through the vacuum of space (astronautical engineering). It sits at the intersection of physics, chemistry, materials science, computer science, and systems thinking, because no single discipline can get a vehicle off the ground alone.</p><p><strong>Why does this field exist?</strong> Humans are not naturally equipped to fly or survive in vacuum, yet flight and spaceflight unlock enormous value: global transportation, satellite communication, weather forecasting, GPS navigation, planetary science, and national security. Every one of those capabilities depends on engineers solving the same brutal problem — how do you push enough mass fast enough, in the right direction, while keeping everything inside alive and functional?</p><p><strong>How the field is organized:</strong> Aerospace engineers typically specialize into a handful of overlapping disciplines: propulsion (how do we generate thrust), structures (how do we build something light enough to fly but strong enough not to break), aerodynamics (how does the vehicle interact with air), guidance-navigation-control or GNC (how does the vehicle know where it is and steer itself), and systems engineering (how do all these pieces get integrated into one working vehicle on schedule and on budget). Over the next several modules, we'll build up each of these one at a time.</p><p><strong>A common misconception</strong> is that aerospace engineering is mostly about building powerful engines. In reality, the discipline is dominated by a much less glamorous obsession: mass. Every kilogram you add anywhere on a vehicle has to be lifted by propellant that itself weighs something, which is why aerospace engineers spend enormous effort removing grams from brackets, wiring, and fasteners. You will see this 'mass obsession' resurface in nearly every module ahead.</p><p><strong>Real-world example:</strong> A modern orbital rocket like the ones used to launch satellites is often over 90% propellant by mass at liftoff — the actual structure, engines, and payload are a thin sliver of the total vehicle. That single fact explains an enormous amount of aerospace engineering decision-making, and we'll return to it constantly.</p>",
        "fact": "The word 'aerospace' itself is a portmanteau — combining 'aeronautics' (flight within the atmosphere) and 'astronautics' (flight beyond it) — reflecting how the field grew from airplanes into rockets over the 20th century.",
        "misconception": "Aerospace engineering is not primarily about building bigger, more powerful engines. It is primarily about managing mass — most of an engineer's decisions trade weight against performance, safety, or cost.",
        "application": "Every satellite that gives you GPS navigation, weather forecasts, or live television, and every crewed or robotic space mission, exists because of the disciplines you're about to learn.",
        "summary": "Aerospace engineering blends propulsion, structures, aerodynamics, and guidance into one integrated system, and nearly every design decision is fundamentally a negotiation over mass.",
        "visualType": "anatomy"
    },
    {
        "id": 2,
        "title": "Module 2: Forces and Newton's Laws",
        "subtitle": "The Physics Foundation Beneath Every Flying Vehicle",
        "badge": "PHYSICS",
        "intro": "Every rocket, airplane, and satellite obeys the same three laws of motion that Isaac Newton wrote down in 1687 — nothing about spaceflight breaks these rules, it just applies them at extreme scale.",
        "lesson": "<p>Before we can talk about rockets, we need the physics vocabulary that describes how anything moves. Newton's First Law says an object stays at rest, or keeps moving in a straight line at constant speed, unless a net force acts on it. In space, where there's essentially no air resistance, this law becomes very literal: a spacecraft coasting with no engine firing will continue in a straight line (or an orbit, which we'll cover soon) indefinitely.</p><p>Newton's Second Law, F = ma, is the single most important equation in this entire curriculum. It tells us that the acceleration of a vehicle depends on the force applied to it divided by its mass. This is why mass reduction matters so much — for the same engine force, a lighter rocket accelerates faster and reaches orbital speed with less propellant.</p><p>Newton's Third Law — for every action there is an equal and opposite reaction — is literally how rocket propulsion works. A rocket engine doesn't 'push against' the air or the ground. It throws mass (hot exhaust gas) out the back at extremely high speed, and the reaction to that expulsion pushes the rocket forward. This is why rockets work perfectly fine in the vacuum of space, where there's nothing to push against except their own exhaust.</p><p><strong>Common misconception:</strong> Many people assume rockets need air to push against, the way a swimmer pushes against water. This is false — rockets actually perform better in vacuum than in atmosphere, because there's no air resistance fighting the exhaust, and no atmospheric pressure pushing back on the nozzle.</p><p><strong>Engineering intuition:</strong> Whenever you see a rocket engineer talk about 'thrust,' they mean the reaction force from expelled mass. Whenever you see them talk about 'weight,' they mean the force of gravity pulling the vehicle down. A rocket lifts off the instant its thrust exceeds its weight.</p>",
        "fact": "Newton's Third Law explains why astronauts on a spacewalk who push against their spacecraft will drift away from it at the same rate they pushed — there's nothing else to stop them in the vacuum of space.",
        "misconception": "Rockets do not need air to 'push against.' They work by expelling mass in one direction, which pushes the vehicle in the opposite direction — and this works better in vacuum than in air.",
        "application": "Engineers use F = ma directly to size engines: given a required acceleration and a vehicle's mass, they can calculate exactly how much thrust an engine must produce.",
        "summary": "Newton's three laws of motion — inertia, F = ma, and action-reaction — are the physics foundation for everything from rocket thrust to orbital coasting.",
        "visualType": "anatomy"
    },
    {
        "id": 3,
        "title": "Module 3: Anatomy of a Rocket",
        "subtitle": "The Major Subsystems That Make Up a Launch Vehicle",
        "badge": "BASICS",
        "intro": "A rocket is not one machine — it's an assembly of four distinct subsystems working in tight coordination: propulsion, structures, guidance and control, and payload.",
        "lesson": "<p>Now that we have the physics, let's look at the physical vehicle itself. A launch vehicle's job is to deliver a payload — a satellite, a crew capsule, a science probe — from the ground into space. To do that, engineers organize the vehicle into four major subsystems.</p><p><strong>Propulsion</strong> includes the engines and the propellant tanks that feed them. This is usually the largest fraction of a rocket's mass and volume, because propellant dominates total vehicle weight (recall Module 1's mass obsession). <strong>Structures</strong> form the physical skeleton and skin of the vehicle — the tanks' walls, the interstage connecting different rocket stages, and the aerodynamic fairing that protects the payload during the trip through the atmosphere. <strong>Guidance, Navigation, and Control (GNC)</strong> is the vehicle's 'brain and inner ear' — flight computers, sensors, and steerable engine nozzles or fins that keep the rocket pointed the right way. <strong>Payload</strong> is whatever the mission actually exists to deliver — everything above is essentially in service of getting this piece to its destination safely.</p><p><strong>Why organize it this way?</strong> Because each subsystem has different design pressures. Propulsion engineers optimize for energy release and combustion efficiency. Structural engineers optimize for strength-to-weight ratio. GNC engineers optimize for precision and reliability under vibration and radiation. Keeping these disciplines conceptually separate lets specialized teams work in parallel — but they must constantly coordinate, because a change in one subsystem (like adding a heavier engine) directly affects the others (more structure needed to support it, more fuel needed to lift it).</p><p><strong>Common misconception:</strong> People often picture a rocket as basically 'an engine with a nose cone.' In reality the engine may be a relatively small fraction of total vehicle length; most of what you see is propellant tankage.</p>",
        "fact": "On many launch vehicles, the propellant tanks' walls are engineered to be only a few millimeters thick — proportionally thinner than the wall of a soda can — because every gram of unnecessary structure steals capacity from the payload.",
        "misconception": "A rocket is not just 'an engine with a nose cone.' Most of its volume and mass is propellant tankage; the engine and payload are comparatively small.",
        "application": "When engineers scope a new rocket program, they start by allocating a 'mass budget' across these four subsystems — exactly the kind of trade-off you'll practice yourself in the Mission Simulation.",
        "summary": "Every rocket is built from four cooperating subsystems — propulsion, structures, guidance/control, and payload — each optimized differently but tightly coupled to the others.",
        "visualType": "anatomy"
    },
    {
        "id": 4,
        "title": "Module 4: Propulsion Systems & Rocket Engines",
        "subtitle": "How Thrust Is Actually Generated",
        "badge": "PROPULSION",
        "intro": "A rocket engine's entire job is to convert stored chemical energy into a high-speed jet of exhaust gas — the faster and more massive that jet, the more thrust the engine produces.",
        "lesson": "<p>We learned in Module 2 that thrust comes from expelling mass. Now let's look at how an engine actually does that. In a chemical rocket engine, fuel and an oxidizer (a chemical that supplies oxygen for combustion, since there's no air in space to burn against) are pumped into a combustion chamber, where they ignite and burn at extremely high temperature and pressure. That hot, high-pressure gas is then forced through a specially shaped nozzle that accelerates it to very high exhaust velocity — often several kilometers per second — before it exits the back of the engine.</p><p><strong>Why the nozzle shape matters:</strong> Rocket nozzles use a 'converging-diverging' (bell) shape. The chamber narrows to a throat, which accelerates the gas to the speed of sound, and then widens again, which — counterintuitively — accelerates the gas further to supersonic speeds as it expands. This shape is carefully tuned to the pressure difference between the chamber and the surrounding environment.</p><p><strong>Engineering intuition — specific impulse:</strong> Engineers measure engine efficiency using a metric called specific impulse (often abbreviated Isp), which essentially describes how much thrust you get per unit of propellant burned per second. A higher specific impulse means an engine gets more 'mileage' out of the same fuel — which directly reduces how much propellant mass a mission needs to carry.</p><p><strong>Common misconception:</strong> Many people assume a 'more powerful' engine is always better. In reality, engineers constantly trade raw thrust (which determines acceleration and liftoff capability) against specific impulse (which determines fuel efficiency) — a high-thrust engine that burns fuel wastefully may be worse for a mission than a gentler, more efficient one, depending on the mission's needs.</p><p><strong>Real-world example:</strong> Engines used for liftoff from Earth prioritize very high thrust to overcome gravity quickly, while engines used for maneuvering in space (where gravity losses matter less) often prioritize efficiency over raw power.</p>",
        "fact": "Rocket engine combustion chambers can reach temperatures hotter than the melting point of the metal they're built from — engineers keep the walls intact using regenerative cooling, circulating cold propellant through channels in the chamber wall before it's burned.",
        "misconception": "A 'more powerful' engine is not automatically a better engine. Thrust (raw power) and specific impulse (fuel efficiency) are separate metrics that engineers must trade off against each other for a given mission.",
        "application": "Mission planners choose engines based on the specific phase of flight — high-thrust engines for liftoff, high-efficiency engines for orbital maneuvers and deep-space cruising.",
        "summary": "Rocket engines generate thrust by accelerating combustion gas through a nozzle, and engineers measure their performance using thrust (raw power) and specific impulse (fuel efficiency) — two metrics that often trade off against each other.",
        "visualType": "anatomy"
    },
    {
        "id": 5,
        "title": "Module 5: Liquid vs. Solid Propellants",
        "subtitle": "Two Fundamentally Different Ways to Store Rocket Fuel",
        "badge": "FUELS",
        "intro": "Rocket propellant comes in two broad families — liquid and solid — and the choice between them is one of the earliest and most consequential decisions in any rocket's design.",
        "lesson": "<p><strong>Solid propellants</strong> are essentially a rubbery, pre-mixed block of fuel and oxidizer cast directly inside the rocket motor casing. Once ignited, a solid rocket motor burns until its propellant is consumed — there is no throttle and, critically, no way to shut it off early. Solid motors are prized for their simplicity, reliability, and the fact that they can sit in storage for years, ready to fire instantly — which is exactly why they're common on military missiles and as strap-on boosters for extra liftoff thrust.</p><p><strong>Liquid propellants</strong> are stored as separate liquid fuel and liquid oxidizer in dedicated tanks, and pumped into the combustion chamber only when needed. Because the flow can be throttled or shut off, liquid engines can be started, stopped, and restarted, and their thrust can be adjusted mid-flight — capabilities essential for precise orbital insertion and landing.</p><p><strong>Why the cryogenic detail matters:</strong> Many high-performance liquid propellants, like liquid hydrogen and liquid oxygen, are gases at room temperature. Engineers chill them to extremely low 'cryogenic' temperatures until they liquefy, because liquids are far denser than gases — this lets far more propellant mass fit into a tank of a given size, directly improving the vehicle's mass budget from Module 1.</p><p><strong>Common misconception:</strong> Students often assume solid propellant is simply the 'weaker' or less advanced option. In reality it's a genuine engineering trade-off: solid motors trade controllability for reliability, simplicity, and storability, while liquid engines trade complexity for precision and efficiency. Neither is strictly superior — the right choice depends entirely on the mission.</p><p><strong>Real-world example:</strong> Many rockets use a hybrid strategy — solid boosters strapped to the sides for a powerful, simple thrust boost during the first minute of flight, combined with a liquid-fueled core stage for the precisely controlled burn that follows.</p>",
        "fact": "Once ignited, a solid rocket motor cannot be turned off — the closest engineers can get to an emergency shutdown is deliberately venting the casing to relieve pressure and stop combustion, a drastic and rarely used measure.",
        "misconception": "Solid propellant is not simply an inferior or outdated technology. It offers reliability, long-term storability, and simplicity that liquid propellant cannot match, which is why it remains standard for boosters and some military systems.",
        "application": "Mission planners select propellant type based on whether a mission needs precise, adjustable thrust (favoring liquids) or maximum simplicity and instant readiness (favoring solids).",
        "summary": "Solid propellants are simple, reliable, and storable but cannot be throttled or shut off; liquid propellants are more complex but allow precise, controllable, restartable thrust — a core trade-off in rocket design.",
        "visualType": "anatomy"
    },
    {
        "id": 6,
        "title": "Module 6: Aerodynamics & Flight Stability",
        "subtitle": "How a Vehicle Interacts With — and Survives — the Atmosphere",
        "badge": "AERO",
        "intro": "Any vehicle passing through air experiences drag, lift, and structural stress — and a rocket must be shaped and controlled carefully to survive its brief but violent trip through the atmosphere.",
        "lesson": "<p>Even though a rocket's ultimate destination may be the vacuum of space, its most dangerous few minutes are often spent inside Earth's atmosphere, where air resistance and mechanical stress are at their worst. Aerodynamics is the study of how air flows around a moving object and the forces that flow creates — primarily <strong>drag</strong> (resistance opposing the direction of motion) and <strong>lift</strong> (a perpendicular force, critical for winged aircraft, less central for vertical rockets but still relevant to fins and control surfaces).</p><p>As a rocket accelerates upward, it experiences a specific, well-known danger point called <strong>Max-Q</strong> — the moment of maximum dynamic pressure, where the combination of increasing speed and still-substantial atmospheric density creates the greatest physical stress on the airframe. Engineers often throttle engines down briefly during Max-Q specifically to reduce structural loads at this critical moment, then throttle back up once the vehicle has climbed into thinner air.</p><p><strong>Flight stability</strong> refers to whether a vehicle naturally tends to correct back toward its intended orientation when disturbed, or tends to tumble further away from it. A rocket's stability depends heavily on where its center of mass sits relative to its center of aerodynamic pressure — engineers carefully arrange components (and sometimes add fins) to keep the vehicle naturally stable, or use active guidance systems to actively correct instability in real time.</p><p><strong>Common misconception:</strong> People often assume rockets are stable simply because they're symmetrical and pointy. In reality, a poorly balanced rocket — even a perfectly symmetric one — can be aerodynamically unstable and tumble uncontrollably without either passive fin stabilization or active guidance correction.</p>",
        "fact": "During Max-Q, a large rocket can experience dynamic pressure loads equivalent to tens of thousands of pounds of force pressing against its skin — engineers design the airframe specifically to survive this narrow window of extreme stress.",
        "misconception": "Symmetry alone does not guarantee stability. A rocket's stability depends on the relationship between its center of mass and its center of aerodynamic pressure, which must be engineered deliberately.",
        "application": "Engineers use throttle control during Max-Q and either fin design or active guidance corrections throughout flight to keep the vehicle both structurally intact and pointed the correct direction.",
        "summary": "Aerodynamic forces like drag and the Max-Q stress peak, combined with the balance between center of mass and center of pressure, determine whether a rocket survives its climb through the atmosphere in a stable, controlled manner.",
        "visualType": "launch"
    },
    {
        "id": 7,
        "title": "Module 7: Guidance, Navigation & Structures",
        "subtitle": "How a Vehicle Knows Where It Is, and What Holds It Together",
        "badge": "GNC",
        "intro": "Guidance, Navigation, and Control (GNC) is the vehicle's sensing and steering brain, while structural engineering is the discipline that keeps the whole assembly from tearing itself apart.",
        "lesson": "<p><strong>Navigation</strong> is the task of figuring out where a vehicle currently is and how fast it's moving, typically using a combination of inertial sensors (which measure acceleration and rotation), star trackers, GPS (near Earth), and ground-based radio tracking. <strong>Guidance</strong> is the task of computing the trajectory the vehicle should follow to reach its target. <strong>Control</strong> is the task of actually commanding engines, thrusters, or control surfaces to follow that guided trajectory. Together, these three form the closed loop — sense, decide, act — that every autonomous or semi-autonomous vehicle relies on.</p><p>Structures engineering, meanwhile, is about ensuring the vehicle physically survives every load it will experience: the crushing weight of stacked stages at liftoff, the vibration of engine combustion, the aerodynamic stress of Max-Q, and the extreme temperature swings of spaceflight. Structural engineers use lightweight, high-strength materials — aluminum-lithium alloys, carbon composites, titanium — and design shapes (like thin-walled cylindrical tanks) that spread stress evenly rather than concentrating it at weak points.</p><p><strong>Why these two disciplines are grouped together here:</strong> GNC and structures are deeply linked, because a structurally flexible rocket (all rockets flex somewhat during flight) can literally confuse guidance sensors or create feedback oscillations if the control system isn't designed to account for it. Engineers must model the vehicle's structural flex directly into the guidance software to avoid dangerous resonance.</p><p><strong>Common misconception:</strong> People often imagine a rocket in flight as a rigid rod. In reality, large rockets flex and bend measurably during flight, and guidance systems are specifically tuned to avoid amplifying that natural flex.</p>",
        "fact": "During the Apollo 11 lunar descent, the guidance computer triggered a '1202' program alarm from sensor data overload — the ground team's fast, correct read of that code (rather than aborting) allowed the landing to proceed safely.",
        "misconception": "A flying rocket is not a rigid rod. Large launch vehicles measurably flex during flight, and guidance and control systems must be tuned to avoid amplifying that structural flex into dangerous oscillation.",
        "application": "GNC systems and structural design are co-engineered so that steering commands never accidentally excite the vehicle's natural flex frequencies.",
        "summary": "Guidance, Navigation, and Control form the sense-decide-act loop that steers a vehicle, while structural engineering ensures it survives physically — and the two disciplines must be designed together to avoid dangerous resonance.",
        "visualType": "team"
    },
    {
        "id": 8,
        "title": "Module 8: Orbits & Orbital Mechanics",
        "subtitle": "Why Objects Stay in Space Once They Get There",
        "badge": "ORBITS",
        "intro": "Reaching space is only half the problem — staying there requires moving sideways fast enough that the ground curves away beneath you as fast as gravity pulls you down.",
        "lesson": "<p>Here's the single most important insight in orbital mechanics: an orbit is not 'floating above gravity' — it's continuous freefall. When a spacecraft reaches roughly 7.8 kilometers per second of horizontal velocity near Earth, gravity pulls it downward at exactly the rate the curved Earth falls away beneath it. The result is a closed loop: the spacecraft is always falling, and always missing the ground, forever (barring atmospheric drag or maneuvers).</p><p>Different orbital altitudes serve different purposes. <strong>Low Earth Orbit (LEO)</strong>, a few hundred kilometers up, is where the International Space Station and most Earth-observation satellites live — it's cheap to reach and offers fast orbital periods (roughly 90 minutes). <strong>Geostationary Earth Orbit (GEO)</strong>, at about 35,786 kilometers, has an orbital period that exactly matches Earth's 24-hour rotation, so a satellite there appears to hover motionless above one point on the equator — ideal for communications and weather satellites.</p><p>To move between orbits, or to travel to another planet, engineers use calculated velocity changes called <strong>Delta-v</strong> maneuvers — essentially, planned engine burns that add or subtract from the spacecraft's velocity at the right moment to shift its trajectory. A <strong>Hohmann transfer</strong> is the most fuel-efficient standard technique for moving between two circular orbits, using two carefully timed burns.</p><p><strong>Escape velocity</strong> is the speed needed to leave a gravitational well entirely rather than settle into orbit around it — roughly 11.2 km/s from Earth's surface. Reaching escape velocity is what's required for missions heading to the Moon, Mars, or beyond, rather than simply orbiting Earth.</p><p><strong>Common misconception:</strong> Many people believe astronauts in orbit are 'weightless' because they're beyond the reach of gravity. In reality, gravity at typical orbital altitudes is only slightly weaker than at the surface — the weightless feeling comes from continuous freefall, not an absence of gravity.</p>",
        "fact": "Because the International Space Station orbits Earth roughly every 90 minutes, its crew experiences about 16 sunrises and sunsets every 24 hours.",
        "misconception": "Astronauts in orbit are not weightless because they've escaped gravity — gravity is still nearly as strong as on the surface. They feel weightless because they, and their spacecraft, are in continuous freefall together.",
        "application": "Mission planners choose orbital altitude and destination based on mission purpose — LEO for observation and crewed stations, GEO for fixed communications coverage, and escape trajectories for interplanetary missions.",
        "summary": "An orbit is continuous freefall around a gravitational body, achieved through sufficient horizontal velocity; Delta-v maneuvers shift between orbits, and escape velocity is required to leave a gravity well entirely.",
        "visualType": "orbit"
    },
    {
        "id": 9,
        "title": "Module 9: Rocket Staging & the Launch Sequence",
        "subtitle": "From Ignition to Orbital Insertion",
        "badge": "LAUNCH",
        "intro": "A launch is a tightly choreographed sequence of physical milestones, and staging — discarding spent rocket sections mid-flight — is what makes reaching orbit possible at all.",
        "lesson": "<p>The launch countdown culminates in ignition, where turbopumps can spin at tens of thousands of RPM to force propellant into the combustion chamber fast enough to sustain full thrust. The instant thrust exceeds vehicle weight, liftoff occurs. Almost immediately, the vehicle performs a <strong>gravity turn</strong> — gently tilting from vertical toward horizontal — which lets gravity itself gradually redirect the rocket's velocity vector rather than forcing an inefficient hard turn later. As the vehicle accelerates and climbs, it passes through Max-Q (Module 6), then continues accelerating as the atmosphere thins.</p><p><strong>Why staging exists</strong> comes directly back to the Tsiolkovsky Rocket Equation, which shows that a rocket's achievable velocity depends on its exhaust velocity and the ratio of its full mass to its empty mass. Carrying an enormous empty fuel tank all the way to orbit is dead weight that wastes energy. Staging solves this: once a stage's propellant is exhausted, engineers jettison that entire empty stage — engines, tank walls, plumbing and all — so the remaining stages only have to accelerate a much lighter vehicle. This is why virtually every orbital rocket in history has used two or more stages.</p><p>After first-stage separation, a second-stage engine ignites (sometimes called SECO for 'second-stage engine cutoff' at the end of its burn) optimized for vacuum operation, continuing acceleration until the vehicle reaches orbital velocity, at which point the payload is released into its planned orbit.</p><p><strong>Common misconception:</strong> Students sometimes assume more stages are always better. In reality, each additional stage adds complexity, separation risk, and extra structural mass at the joints — engineers optimize the number of stages for the specific mission rather than maximizing it.</p>",
        "fact": "At liftoff, a large orbital rocket can burn tons of propellant every second — a Saturn V, for instance, consumed roughly 20 tons of propellant per second at full thrust.",
        "misconception": "More rocket stages are not automatically better. Each stage adds separation risk, structural mass, and complexity, so engineers choose the minimum number of stages that achieves the mission.",
        "application": "Launch vehicle designers balance stage count, gravity-turn timing, and throttle profiles (including throttling down at Max-Q) to maximize payload delivered to orbit while minimizing risk.",
        "summary": "A launch proceeds through ignition, a gravity turn, Max-Q, and staged separation — where jettisoning spent stages removes dead weight so the rocket equation works in the vehicle's favor all the way to orbit.",
        "visualType": "launch"
    },
    {
        "id": 10,
        "title": "Module 10: Space Weather & the Mission Environment",
        "subtitle": "The Invisible Hazards Beyond Earth's Atmosphere",
        "badge": "WEATHER",
        "intro": "Space is not a calm, empty vacuum — it is constantly bombarded by high-energy solar radiation and magnetic disturbances that can damage electronics and endanger astronauts.",
        "lesson": "<p><strong>Solar flares</strong> are sudden, intense bursts of electromagnetic radiation (X-rays and ultraviolet light) released from the Sun's surface, traveling at the speed of light and reaching Earth in about eight minutes. <strong>Coronal Mass Ejections (CMEs)</strong> are separate, slower-moving events where the Sun ejects billions of tons of magnetized plasma into interplanetary space, typically arriving at Earth over one to three days.</p><p>Both phenomena threaten spacecraft in similar but distinct ways. Energetic particles can cause <strong>Single Event Upsets (SEUs)</strong> — essentially a stray high-energy particle flipping a bit in a microchip's memory, corrupting data or commands. Larger events can induce electrical currents strong enough to damage power systems outright, and unshielded astronauts face genuine radiation-exposure risk during major solar events.</p><p><strong>Why engineers care so much:</strong> This is exactly why space missions carry radiation shielding — typically layers of aluminum, polyethylene, or specialized composites — and why organizations like NASA operate space weather monitoring systems (such as the DONKI database this platform uses) to provide early warning of solar activity, allowing mission controllers to power down sensitive systems or delay spacewalks during high-risk windows.</p><p><strong>Common misconception:</strong> People often imagine radiation shielding as simply 'more metal is always better.' In reality, some shielding materials can make certain types of radiation worse by generating secondary particle showers on impact — engineers select shielding material and thickness carefully based on the specific radiation environment of the mission's destination and duration.</p><p><strong>Real-world example:</strong> The 1859 Carrington Event, the most intense geomagnetic storm on record, was powerful enough to spark widespread telegraph malfunctions and produce aurora visible near the equator — a reminder that a similarly sized event today could seriously threaten modern satellite infrastructure.</p>",
        "fact": "NASA's DONKI (Space Weather Database Of Notifications, Knowledge, Information) system — which powers this platform's live weather feed — continuously tracks solar flares and CMEs to give mission planners early warning.",
        "misconception": "More radiation shielding is not automatically safer. Certain thick, dense materials can generate secondary particle showers on impact, so engineers must match shielding material and thickness to the specific radiation environment of the mission.",
        "application": "Mission planners use live space weather data (like the feed in this platform's simulator) to decide whether protection budgets need to increase, or whether a launch or spacewalk should be delayed.",
        "summary": "Solar flares and coronal mass ejections pose distinct but serious risks to spacecraft electronics and astronaut safety, and engineers use monitoring systems like NASA's DONKI along with carefully chosen shielding to manage that risk.",
        "visualType": "weather"
    }
]

ACADEMY_1_QUESTIONS = {
    1: [
        {
            "question": "What is the central, defining obsession that drives most aerospace engineering design decisions?",
            "options": ["Building the most powerful engine possible", "Minimizing mass everywhere on the vehicle", "Making the vehicle as aerodynamic as possible", "Maximizing the number of onboard sensors"],
            "correct": 1,
            "reasoning": "Because every kilogram of mass must be lifted by propellant that itself has mass, aerospace engineers relentlessly focus on removing unnecessary weight from every subsystem."
        },
        {
            "question": "Which subsystem grouping best describes how a launch vehicle is typically organized?",
            "options": ["Engine, wings, wheels, and cabin", "Propulsion, structures, guidance/control, and payload", "Fuel, oxygen, water, and battery", "Radar, radio, camera, and antenna"],
            "correct": 1,
            "reasoning": "Aerospace vehicles are organized into propulsion, structures, guidance-navigation-control, and payload subsystems, each with different design priorities."
        },
        {
            "question": "Roughly what fraction of a typical orbital rocket's liftoff mass is propellant?",
            "options": ["About 10%", "About 50%", "Over 90%", "Exactly 25%"],
            "correct": 2,
            "reasoning": "Orbital rockets are often more than 90% propellant by mass at liftoff, which is why mass reduction elsewhere on the vehicle is so critical."
        },
        {
            "question": "What does the term 'astronautics' refer to, as distinct from 'aeronautics'?",
            "options": ["Flight within Earth's atmosphere", "Flight beyond the atmosphere, in space", "The study of astronaut training only", "Weather balloon operations"],
            "correct": 1,
            "reasoning": "Aeronautics covers atmospheric flight, while astronautics covers spaceflight — together they form the word 'aerospace.'"
        },
        {
            "question": "Why does aerospace engineering require multiple specialized disciplines working together?",
            "options": ["Because government regulations require it", "Because no single discipline can solve propulsion, structural, aerodynamic, and guidance problems alone", "Because it creates more jobs", "Because rockets are built in multiple countries"],
            "correct": 1,
            "reasoning": "Getting a vehicle into space requires solving distinct physics and engineering problems simultaneously, which is why the field integrates propulsion, structures, aerodynamics, and GNC specialists."
        }
    ],
    2: [
        {
            "question": "According to Newton's First Law, what happens to a coasting spacecraft with no engine firing and no external forces?",
            "options": ["It gradually slows to a stop", "It continues in a straight line at constant velocity", "It automatically enters orbit", "It accelerates due to residual thrust"],
            "correct": 1,
            "reasoning": "Newton's First Law states an object in motion stays in motion at constant velocity unless acted on by a net force — in the near-vacuum of space, this holds very literally."
        },
        {
            "question": "What does Newton's Second Law (F = ma) tell engineers about reducing a rocket's mass?",
            "options": ["Reducing mass has no effect on acceleration", "For the same force, reducing mass increases acceleration", "Reducing mass always reduces thrust", "Mass and acceleration are unrelated in spaceflight"],
            "correct": 1,
            "reasoning": "F = ma means acceleration equals force divided by mass — so for a fixed engine force, a lighter vehicle accelerates faster."
        },
        {
            "question": "Why do rocket engines work in the vacuum of space, where there is no air to push against?",
            "options": ["Rockets actually rely on residual atmospheric pressure at all times", "Newton's Third Law means the reaction to expelled exhaust mass pushes the rocket forward regardless of surrounding air", "Rockets use magnetic propulsion in vacuum", "They don't actually work in vacuum without air"],
            "correct": 1,
            "reasoning": "Rocket thrust comes from expelling mass (exhaust) and receiving an equal, opposite reaction force — this works with or without surrounding air, and actually works better in vacuum."
        },
        {
            "question": "What is a common misconception about how rocket propulsion works?",
            "options": ["That rockets need air to push against, like a swimmer pushing against water", "That rockets use liquid fuel", "That rockets have engines", "That rockets experience gravity"],
            "correct": 0,
            "reasoning": "Rockets do not push against air — they work by expelling their own mass, which is why they perform even better in the vacuum of space than in the atmosphere."
        },
        {
            "question": "What determines the moment a rocket lifts off the launch pad?",
            "options": ["When the countdown clock reaches zero", "The instant engine thrust exceeds the vehicle's weight", "When all fuel has been loaded", "When ground crews manually release clamps regardless of thrust"],
            "correct": 1,
            "reasoning": "Liftoff occurs physically the moment upward thrust force exceeds the downward force of gravity (weight) acting on the vehicle."
        }
    ],
    3: [
        {
            "question": "Which of the following is one of the four major subsystems that make up a launch vehicle?",
            "options": ["Guidance, Navigation, and Control (GNC)", "Marketing and Public Relations", "Weather Forecasting", "Ground Transportation"],
            "correct": 0,
            "reasoning": "GNC is one of the four major rocket subsystems, alongside propulsion, structures, and payload."
        },
        {
            "question": "Why do propellant tanks typically make up the largest fraction of a rocket's mass and volume?",
            "options": ["Because tanks are made of very heavy materials by design", "Because propellant dominates total vehicle weight, as covered in Module 1", "Because engines require large tanks to operate at all", "Because regulations require oversized tanks"],
            "correct": 1,
            "reasoning": "Since propellant is often over 90% of liftoff mass, the tanks that hold it necessarily make up the largest portion of the vehicle's structure and volume."
        },
        {
            "question": "What is a common misconception about a rocket's physical appearance?",
            "options": ["That it has a payload", "That it is basically 'an engine with a nose cone,' when in fact most of its volume is propellant tankage", "That it has fins", "That it launches vertically"],
            "correct": 1,
            "reasoning": "The engine is often a relatively small fraction of a rocket's total structure — the bulk of the vehicle is propellant tankage."
        },
        {
            "question": "Why must the four rocket subsystems constantly coordinate with each other during design?",
            "options": ["Because they are legally required to communicate", "Because a change in one subsystem, like a heavier engine, directly affects mass and structural requirements in the others", "Because they share the same physical location only", "They do not need to coordinate"],
            "correct": 1,
            "reasoning": "Since subsystems are physically and functionally linked, a design change in one (like engine mass) ripples into requirements for structures and propellant capacity in the others."
        },
        {
            "question": "What is the primary purpose of the payload subsystem on a launch vehicle?",
            "options": ["To generate thrust", "To provide structural support for the fuel tanks", "To carry the satellite, science instrument, or crew capsule that the mission exists to deliver", "To steer the rocket during ascent"],
            "correct": 2,
            "reasoning": "The payload is the actual cargo the mission is designed to deliver — everything else on the vehicle exists in service of getting the payload to its destination."
        }
    ],
    4: [
        {
            "question": "What is the basic function of a rocket engine's nozzle?",
            "options": ["To cool the combustion chamber only", "To accelerate hot combustion gas to high exhaust velocity before it exits the engine", "To mix fuel and oxidizer before ignition", "To store propellant"],
            "correct": 1,
            "reasoning": "The nozzle's converging-diverging shape accelerates combustion gas to high supersonic exhaust velocities, which is what generates thrust."
        },
        {
            "question": "What does 'specific impulse' (Isp) measure in a rocket engine?",
            "options": ["The total thrust an engine can produce at liftoff", "How efficiently an engine converts propellant into thrust over time", "The temperature of the combustion chamber", "The physical size of the engine"],
            "correct": 1,
            "reasoning": "Specific impulse measures propellant efficiency — how much thrust is generated per unit of propellant consumed per second, essentially the engine's 'fuel mileage.'"
        },
        {
            "question": "Why can't rocket engine designers simply maximize both thrust and specific impulse simultaneously?",
            "options": ["Because thrust and efficiency often trade off against each other depending on engine design and mission needs", "Because it is against international regulations", "Because thrust and specific impulse are the same measurement", "Because only solid engines can have high thrust"],
            "correct": 0,
            "reasoning": "High-thrust engines and high-efficiency engines often involve different design choices, so engineers select the balance appropriate to a specific mission phase."
        },
        {
            "question": "Why do engines used for liftoff from Earth typically prioritize high thrust over maximum efficiency?",
            "options": ["Because efficiency doesn't matter in any phase of flight", "Because overcoming Earth's gravity quickly requires substantial raw thrust", "Because high-thrust engines always use less fuel", "Because liftoff engines don't need combustion"],
            "correct": 1,
            "reasoning": "Liftoff must overcome significant gravity losses quickly, so raw thrust is prioritized, whereas in-space maneuvering engines can prioritize fuel efficiency instead."
        },
        {
            "question": "How do engineers keep a combustion chamber from melting despite temperatures exceeding its material's melting point?",
            "options": ["They use regenerative cooling, circulating cold propellant through the chamber walls before combustion", "They only fire the engine for a fraction of a second", "Combustion chambers never actually get that hot", "They coat the chamber in ice before launch"],
            "correct": 0,
            "reasoning": "Regenerative cooling routes propellant through channels in the chamber wall to absorb heat before it is burned, keeping the structure intact despite extreme internal temperatures."
        }
    ],
    5: [
        {
            "question": "What is the key operational limitation of a solid rocket motor once ignited?",
            "options": ["It can be throttled freely", "It cannot be shut off or restarted once burning", "It requires liquid oxygen to function", "It can only be used in space, never at liftoff"],
            "correct": 1,
            "reasoning": "Solid motors burn continuously once ignited and cannot be throttled or shut down like liquid engines can."
        },
        {
            "question": "Why are propellants like liquid hydrogen and liquid oxygen cooled to cryogenic temperatures?",
            "options": ["To prevent them from ever combusting accidentally", "To liquefy gases into a much denser state, allowing more propellant mass to fit in a given tank volume", "Because cold propellant burns hotter", "To reduce the cost of the propellant"],
            "correct": 1,
            "reasoning": "Cooling propellants until they liquefy dramatically increases their density, letting engineers pack more propellant mass into smaller, lighter tanks."
        },
        {
            "question": "What is a key misconception about solid rocket propellant?",
            "options": ["That it is simply an inferior, outdated technology compared to liquid propellant", "That it is used in boosters", "That it can be stored", "That it involves an oxidizer"],
            "correct": 0,
            "reasoning": "Solid propellant is not inferior — it trades controllability for simplicity, reliability, and long-term storability, making it the right choice for specific missions."
        },
        {
            "question": "Why do liquid-fueled engines offer capabilities that solid motors cannot?",
            "options": ["Liquid engines can be throttled, shut down, and restarted mid-flight, enabling precise orbital insertion and landing", "Liquid engines are always cheaper", "Liquid engines never require an oxidizer", "Liquid engines are simpler to build"],
            "correct": 0,
            "reasoning": "Because liquid propellant flow can be controlled, liquid engines support throttling, shutdown, and restart — essential for precision maneuvers that solid motors cannot perform."
        },
        {
            "question": "Why do many rockets combine solid strap-on boosters with a liquid-fueled core stage?",
            "options": ["To combine simple, powerful initial thrust from solids with the precise, efficient control of liquids for the rest of the flight", "Because regulations require both fuel types", "Because solid fuel is required for orbital insertion", "Because liquid engines cannot generate enough thrust alone under any circumstance"],
            "correct": 0,
            "reasoning": "This hybrid strategy captures the strengths of both propellant types: solids for a powerful, simple initial boost, and liquids for the precisely controlled burn that follows."
        }
    ],
    6: [
        {
            "question": "What does the term 'Max-Q' refer to during a rocket's ascent?",
            "options": ["The maximum speed of the vehicle", "The point of maximum dynamic pressure, and thus maximum structural stress from the atmosphere", "The moment of engine ignition", "The maximum altitude reached"],
            "correct": 1,
            "reasoning": "Max-Q is the point where the combination of vehicle speed and atmospheric density creates the greatest mechanical stress on the airframe."
        },
        {
            "question": "Why do engineers often throttle engines down briefly during Max-Q?",
            "options": ["To save fuel for later in the mission", "To reduce structural loads on the airframe during the period of maximum stress", "Because the engine automatically shuts off at that altitude", "To allow the guidance computer to reboot"],
            "correct": 1,
            "reasoning": "Reducing thrust briefly during Max-Q lowers acceleration and dynamic pressure, protecting the vehicle from structural failure during its most stressful moment."
        },
        {
            "question": "What determines whether a rocket is aerodynamically stable during flight?",
            "options": ["Whether it is painted a certain color", "The relationship between its center of mass and its center of aerodynamic pressure", "Its total height only", "The number of engines it has"],
            "correct": 1,
            "reasoning": "Stability depends on how the center of mass relates to the center of aerodynamic pressure — this relationship, not symmetry alone, determines whether a vehicle naturally corrects or tumbles."
        },
        {
            "question": "What is a common misconception about rocket stability?",
            "options": ["That a symmetrical, pointy shape is automatically stable", "That fins can help stability", "That guidance systems can actively correct instability", "That center of mass matters"],
            "correct": 0,
            "reasoning": "Symmetry does not guarantee stability — a poorly balanced rocket can still tumble unless its center of mass and center of pressure are properly engineered."
        },
        {
            "question": "Which two aerodynamic forces are most central to how a vehicle interacts with the air it moves through?",
            "options": ["Drag and lift", "Voltage and current", "Torque and inertia", "Pressure and temperature only"],
            "correct": 0,
            "reasoning": "Drag (resistance opposing motion) and lift (a perpendicular force) are the fundamental aerodynamic forces engineers must account for."
        }
    ],
    7: [
        {
            "question": "What are the three components of the Guidance, Navigation, and Control (GNC) loop?",
            "options": ["Sense, decide, act — corresponding to navigation, guidance, and control", "Launch, orbit, land", "Fuel, oxidizer, ignition", "Radio, radar, camera"],
            "correct": 0,
            "reasoning": "Navigation determines position/velocity (sense), guidance computes the needed trajectory (decide), and control commands the actuators to follow it (act)."
        },
        {
            "question": "Why must structural engineers use lightweight, high-strength materials like aluminum-lithium alloys or carbon composites?",
            "options": ["To meet aesthetic design requirements", "To survive extreme mechanical and thermal loads without adding excessive mass", "Because heavier materials are cheaper", "Because these materials are required by international law"],
            "correct": 1,
            "reasoning": "Structural materials must combine strength with low mass, since every kilogram of unnecessary structure reduces available payload capacity."
        },
        {
            "question": "Why must guidance and control systems account for a rocket's structural flex during flight?",
            "options": ["Because a rigid rocket never flexes and this is irrelevant", "Because steering commands could otherwise excite the vehicle's natural flex frequencies, causing dangerous oscillation", "Because flex only matters after engine cutoff", "Because it improves fuel efficiency"],
            "correct": 1,
            "reasoning": "Large rockets flex measurably during flight, and improperly tuned control systems can amplify that flex into dangerous resonant oscillation, so GNC and structural design must be co-engineered."
        },
        {
            "question": "What was significant about the Apollo 11 '1202' program alarm during lunar descent?",
            "options": ["It indicated a fuel leak", "It signaled sensor data overload, and the ground team's fast, correct assessment allowed the landing to proceed safely", "It was a structural failure warning", "It required an immediate mission abort"],
            "correct": 1,
            "reasoning": "The 1202 alarm indicated the guidance computer was overloaded with sensor data; correctly diagnosing it as non-critical allowed the landing to continue instead of aborting."
        },
        {
            "question": "What is a common misconception about a rocket's structure during flight?",
            "options": ["That it is a perfectly rigid rod that never bends", "That it uses lightweight materials", "That it experiences stress at liftoff", "That guidance systems monitor it"],
            "correct": 0,
            "reasoning": "Large rockets are not perfectly rigid — they measurably flex during flight, which guidance and control systems must be designed to accommodate."
        }
    ],
    8: [
        {
            "question": "What is the fundamental physical description of an orbit?",
            "options": ["A location beyond the reach of gravity", "A state of continuous freefall where forward velocity matches the rate the ground curves away", "A fixed point held up by rocket thrust", "A region with no gravitational pull"],
            "correct": 1,
            "reasoning": "An orbit is continuous freefall — the spacecraft is always falling toward the planet, but moving sideways fast enough that the surface curves away at the same rate."
        },
        {
            "question": "What distinguishes Geostationary Earth Orbit (GEO) from Low Earth Orbit (LEO)?",
            "options": ["GEO is inside the atmosphere while LEO is not", "GEO's orbital period matches Earth's rotation, so a satellite appears fixed above one point on the equator", "LEO satellites move slower than GEO satellites", "GEO requires no orbital velocity at all"],
            "correct": 1,
            "reasoning": "At GEO altitude (~35,786 km), a satellite's orbital period exactly matches Earth's 24-hour rotation, making it appear stationary above a fixed point."
        },
        {
            "question": "What is a Hohmann transfer used for?",
            "options": ["Communicating with ground control", "Moving between two circular orbits using the most fuel-efficient standard two-burn technique", "Cooling the spacecraft's electronics", "Measuring space weather"],
            "correct": 1,
            "reasoning": "A Hohmann transfer is a standard, fuel-efficient method for moving a spacecraft between two orbits using two precisely timed engine burns."
        },
        {
            "question": "What is a common misconception about why astronauts feel weightless in orbit?",
            "options": ["That they have escaped Earth's gravity entirely", "That they are in continuous freefall alongside their spacecraft, even though gravity is still nearly full strength", "That weightlessness is caused by lack of air", "That weightlessness only happens near the Moon"],
            "correct": 0,
            "reasoning": "Gravity at typical orbital altitudes is only slightly weaker than at the surface — weightlessness comes from continuous freefall, not distance from gravity."
        },
        {
            "question": "What is escape velocity?",
            "options": ["The speed needed to enter a stable orbit", "The speed needed to leave a gravitational well entirely, rather than settle into orbit", "The maximum speed a rocket engine can produce", "The speed of sound at sea level"],
            "correct": 1,
            "reasoning": "Escape velocity (about 11.2 km/s from Earth's surface) is the speed required to break free of a gravitational well entirely, necessary for missions beyond Earth orbit."
        }
    ],
    9: [
        {
            "question": "What is the purpose of a 'gravity turn' immediately after liftoff?",
            "options": ["To test the guidance computer", "To let gravity gradually redirect the rocket's velocity from vertical toward horizontal, rather than forcing an inefficient hard turn later", "To reduce engine temperature", "To separate the first stage early"],
            "correct": 1,
            "reasoning": "A gravity turn uses gravity itself to gradually tilt the rocket's trajectory from vertical to horizontal, which is far more efficient than an abrupt directional change."
        },
        {
            "question": "According to the Tsiolkovsky Rocket Equation, why is staging necessary for reaching orbit?",
            "options": ["Because international law requires multiple stages", "Because carrying empty, spent propellant tanks all the way to orbit wastes enormous energy, so jettisoning them improves the mass ratio", "Because engines cannot fire more than once", "Because payloads must be split across stages"],
            "correct": 1,
            "reasoning": "The rocket equation shows that discarding dead weight (spent tanks and engines) during flight dramatically improves the achievable velocity of the remaining vehicle."
        },
        {
            "question": "What is a common misconception about the number of stages a rocket should have?",
            "options": ["That more stages are always better", "That staging exists at all", "That gravity turns are useful", "That Max-Q occurs during ascent"],
            "correct": 0,
            "reasoning": "More stages are not always better — each additional stage adds separation risk, structural mass, and complexity, so engineers optimize stage count for the specific mission."
        },
        {
            "question": "What typically happens immediately after first-stage separation on a multi-stage rocket?",
            "options": ["The mission is aborted", "A second-stage engine, optimized for vacuum operation, ignites to continue acceleration toward orbital velocity", "The payload is immediately released", "The rocket begins descent back to the pad"],
            "correct": 1,
            "reasoning": "After the first stage separates, a vacuum-optimized second-stage engine ignites and continues accelerating the lighter remaining vehicle toward orbital velocity."
        },
        {
            "question": "Roughly how much propellant might a large rocket like a Saturn V consume per second at full liftoff thrust?",
            "options": ["About 20 tons per second", "About 2 kilograms per second", "About 200 grams per second", "None — solid rockets don't consume propellant continuously"],
            "correct": 0,
            "reasoning": "Large liquid rockets can burn tons of propellant per second at full thrust; the Saturn V, for example, burned roughly 20 tons per second at liftoff."
        }
    ],
    10: [
        {
            "question": "What is the key difference between a solar flare and a Coronal Mass Ejection (CME)?",
            "options": ["They are the same phenomenon with different names", "A solar flare is a fast burst of electromagnetic radiation, while a CME is a slower-moving cloud of magnetized plasma", "CMEs travel faster than flares", "Solar flares only affect Earth's oceans"],
            "correct": 1,
            "reasoning": "Solar flares are radiation bursts arriving at light speed, while CMEs are slower-moving clouds of charged plasma that typically take one to three days to reach Earth."
        },
        {
            "question": "What is a Single Event Upset (SEU) in spacecraft electronics?",
            "options": ["A permanent hardware failure caused by heat", "A stray high-energy particle flipping a bit in a microchip's memory, corrupting data or commands", "A software bug unrelated to radiation", "A structural crack caused by vibration"],
            "correct": 1,
            "reasoning": "An SEU occurs when a high-energy particle strikes a microchip and flips a memory bit, potentially corrupting data or commands without physically destroying the hardware."
        },
        {
            "question": "What is a common misconception about radiation shielding for spacecraft?",
            "options": ["That more shielding material is always safer", "That shielding uses aluminum or polyethylene", "That NASA monitors space weather", "That CMEs pose a risk to astronauts"],
            "correct": 0,
            "reasoning": "More shielding is not automatically safer — some dense materials can generate secondary particle showers on impact, so material and thickness must be matched to the specific radiation environment."
        },
        {
            "question": "What is the purpose of NASA's DONKI system, used in this platform's live weather feed?",
            "options": ["To track satellite manufacturing costs", "To monitor and provide early warning of solar flares and CMEs for mission planners", "To schedule astronaut meals", "To calculate rocket staging sequences"],
            "correct": 1,
            "reasoning": "DONKI (Space Weather Database Of Notifications, Knowledge, Information) tracks solar activity to give mission planners advance warning of hazardous space weather."
        },
        {
            "question": "What made the 1859 Carrington Event historically significant?",
            "options": ["It was the first crewed spaceflight", "It was an extremely intense geomagnetic storm that disrupted telegraph systems and produced aurora near the equator", "It destroyed the first artificial satellite", "It caused a rocket launch failure"],
            "correct": 1,
            "reasoning": "The Carrington Event was the most intense recorded geomagnetic storm, causing widespread telegraph disruption and aurora visible far from the poles — a benchmark for assessing modern space weather risk."
        }
    ]
}


# ====================================================================
# ROCKET ACADEMY II — MISSION PLANNING ACADEMY
# Ten modules that directly teach every concept and parameter used
# inside the Mission Simulation capstone experience.
# ====================================================================

ACADEMY_2_MODULES = [
    {
        "id": 1,
        "title": "Module 1: Mission Objectives & Systems Thinking",
        "subtitle": "Defining What a Mission Must Achieve Before Designing Anything",
        "badge": "OBJECTIVES",
        "intro": "Every real mission begins not with a rocket design, but with a clearly defined objective — and every downstream engineering decision exists to serve that objective.",
        "lesson": "<p>Before an engineer touches a budget slider or picks an engine, a mission needs a clear objective: what are we trying to accomplish, and how will we know if we succeeded? A communications satellite mission succeeds by reaching a stable orbit and operating for years. A sample-return mission succeeds only if material physically comes back to Earth intact. These very different definitions of success lead to very different engineering priorities.</p><p><strong>Systems thinking</strong> is the discipline of recognizing that a spacecraft is not a collection of independent parts, but a tightly interconnected system where every decision ripples outward. Add radiation shielding, and you add mass, which demands more propellant, which demands a bigger engine, which adds more mass — this loop is why experienced mission planners think in terms of the whole system rather than optimizing one part in isolation.</p><p>In the Mission Simulation you're about to use, every parameter you configure — destination, budget, engine allocation, protection allocation, payload allocation — is a systems-thinking exercise in miniature. Changing one slider changes what the others can realistically support.</p><p><strong>Common misconception:</strong> Students often assume there's a single 'correct' configuration that maximizes every metric at once. In real engineering, nearly every decision is a trade-off — more of one desirable thing generally means less of another.</p>",
        "fact": "NASA's James Webb Space Telescope required over two decades of systems-level trade-off work to balance its 6.5-meter mirror against the size constraints of its Ariane 5 launch vehicle fairing.",
        "misconception": "There is rarely a single 'correct' mission configuration that maximizes every metric simultaneously. Nearly every design decision trades one advantage against another.",
        "application": "The Mission Simulation directly tests this skill — you will define an objective (destination), then allocate limited resources across competing priorities, exactly as real mission planners do.",
        "summary": "Every mission starts with a clear objective, and systems thinking — recognizing how every subsystem decision affects the others — is the foundation of all mission planning that follows.",
        "visualType": "team"
    },
    {
        "id": 2,
        "title": "Module 2: Destination Selection & Trajectory Trade-offs",
        "subtitle": "Why 'Where You're Going' Drives Almost Everything Else",
        "badge": "DESTINATION",
        "intro": "The destination a mission targets — Mars, the Moon, an icy moon like Europa, or the asteroid belt — fundamentally determines the Delta-v budget, travel time, and environmental hazards the whole mission must plan around.",
        "lesson": "<p>Destination is the very first parameter you'll set in the Mission Simulation, and it isn't just a label — it drives nearly every other engineering requirement. Reaching different destinations requires very different amounts of Delta-v (recall Rocket Academy I, Module 8): a lunar mission requires meaningfully less propulsive effort than an interplanetary journey to Mars, and a mission to an outer icy moon like Europa demands substantially more still.</p><p>Destination also determines the mission's environment. A lunar surface mission must deal with abrasive regolith dust and extreme day-night temperature swings but comparatively little radiation shielding demand relative to interplanetary space. A Mars mission spends months in interplanetary space exposed to solar radiation before ever arriving. An icy moon mission may need to consider Jupiter's intense radiation belts. An asteroid belt mission introduces micrometeorite and navigation-precision challenges around irregular, weakly-gravitating bodies.</p><p><strong>Why this matters for your simulation:</strong> When you later see the AI Tutor evaluate your protection or payload allocation as too low or too high, it's implicitly weighing your destination's specific hazards. A budget that's perfectly safe for a lunar mission might be dangerously under-protected for a deep-space Europa mission.</p><p><strong>Common misconception:</strong> Students often treat 'destination' as a purely cosmetic choice. In real mission planning, destination selection is often the single decision from which every other budget allocation is derived.</p>",
        "fact": "A trip to Mars typically takes six to nine months one-way using efficient transfer trajectories, while a trip to the Moon takes only a few days — a difference that dramatically changes radiation exposure and life-support requirements.",
        "misconception": "Destination is not a cosmetic label. It is the foundational decision that determines Delta-v requirements, radiation exposure, and structural demands for the rest of the mission.",
        "application": "In the simulator, your destination choice should directly inform how you allocate your protection and propulsion budgets — a Europa mission generally warrants a very different balance than a lunar mission.",
        "summary": "Destination determines Delta-v needs, travel time, and environmental hazards, making it the foundational decision from which most other mission-planning choices flow.",
        "visualType": "orbit"
    },
    {
        "id": 3,
        "title": "Module 3: Payload Planning & Science Trade-offs",
        "subtitle": "Deciding What the Mission Actually Carries",
        "badge": "PAYLOAD",
        "intro": "Payload is the reason the mission exists, but every gram allocated to instruments or cargo is a gram unavailable to propulsion or protection — making payload planning a constant balancing act.",
        "lesson": "<p>Payload is whatever the mission is actually built to deliver or carry out its work with: scientific instruments, cameras, communications equipment, sample-collection tools, or crew life-support systems. It's tempting to think 'more payload is always better,' since payload capability is often what defines mission success. But payload mass draws directly from the same overall mass budget as propulsion and protection.</p><p>In the Mission Simulation, your <strong>Science Payloads</strong> percentage represents how much of your total mission budget is dedicated to payload capability. A higher payload allocation generally increases the scientific or operational value of the mission — but it comes at the direct expense of budget available for engine performance or protective shielding.</p><p><strong>Engineering intuition:</strong> Experienced mission planners ask, 'what is the minimum payload capability that still achieves the mission objective?' rather than 'how much payload can we possibly fit?' Overbuilding payload capacity at the expense of propulsion or protection can leave a mission unable to reach its destination safely, or unable to survive the environment once it arrives — value delivered by a payload that never survives to use it is zero.</p><p><strong>Common misconception:</strong> Assuming that maximizing payload budget always maximizes mission value. In practice, an underpowered or under-protected spacecraft with an enormous payload allocation may fail before that payload ever does anything useful.</p>",
        "fact": "The James Webb Space Telescope's science payload — its mirror and instruments — required a sunshield the size of a tennis court simply to keep the payload cold enough to function, illustrating how protection and payload needs are deeply intertwined.",
        "misconception": "Maximizing payload allocation does not automatically maximize mission value. A spacecraft that can't survive the trip or the environment delivers zero value from its payload, however capable that payload might be.",
        "application": "In the simulator, treat your Science Payloads slider as 'how much capability do I need to achieve the objective,' not 'how much can I possibly fit' — balance it against propulsion and protection.",
        "summary": "Payload is the mission's purpose, but it draws from the same limited budget as propulsion and protection, so effective planning asks what payload capability is truly necessary rather than maximal.",
        "visualType": "anatomy"
    },
    {
        "id": 4,
        "title": "Module 4: Budget Allocation & Engineering Trade-offs",
        "subtitle": "The Central Skill Tested Inside the Mission Simulation",
        "badge": "BUDGET",
        "intro": "An engineering trade-off is the act of balancing competing constraints — mass, power, and cost — where improving one system typically requires giving something up elsewhere.",
        "lesson": "<p>This module is the conceptual heart of the entire Mission Simulation. A mission's <strong>total budget</strong> (in millions of dollars) is a fixed resource that must be distributed across three competing priorities: engine propulsion, protection and shielding, and science payloads. Every additional percentage point allocated to one of these necessarily comes from the others — this is the definition of an engineering trade-off.</p><p><strong>Why trade-offs are unavoidable:</strong> A fixed total budget means propulsion, protection, and payload are in direct competition. Allocate too much to propulsion, and you may have a fast, well-protected mission with very limited scientific value. Allocate too much to payload, and you may have an incredibly capable instrument suite riding on an underpowered, poorly shielded spacecraft that never safely arrives. There is no configuration that maximizes all three simultaneously — the entire discipline of mission planning is finding the balance appropriate to the specific mission's objective and destination.</p><p><strong>Engineering intuition:</strong> Rather than asking 'what's the best possible allocation,' experienced planners ask 'what does this specific destination, on this specific budget, actually require to succeed?' A lunar mission with modest radiation risk can typically afford to allocate more toward payload than a deep-space Europa mission facing intense radiation exposure.</p><p><strong>Common misconception:</strong> Assuming a 'balanced' 33/33/33 split across the three sliders is always optimal. The right balance depends entirely on destination, objective, and the mission's specific risk profile — sometimes an intentionally unbalanced allocation is the correct engineering choice.</p>",
        "fact": "NASA's Perseverance rover mission spent years in trade studies simply balancing how much mass to dedicate to its sampling system versus its power system versus its scientific instruments before finalizing a design.",
        "misconception": "An even, balanced split across propulsion, protection, and payload is not automatically the optimal configuration. The right allocation depends entirely on the specific destination and mission objective.",
        "application": "This module maps directly onto the three core sliders in the Mission Simulation — Engine Propulsion, Protection & Shielding, and Science Payloads — which together must sum to your available budget.",
        "summary": "Budget allocation across propulsion, protection, and payload is the central engineering trade-off in mission planning, and the right balance depends on destination and mission objective rather than any single universal formula.",
        "visualType": "tradeoff"
    },
    {
        "id": 5,
        "title": "Module 5: Engine Selection & Delta-v Budgets",
        "subtitle": "Translating 'Engine Propulsion %' Into Real Mission Capability",
        "badge": "PROPULSION",
        "intro": "Your Engine Propulsion budget determines how much Delta-v capability your spacecraft has available — and different destinations demand very different amounts of it.",
        "lesson": "<p>Recall from Rocket Academy I that Delta-v is the total velocity change a spacecraft can achieve using its available propellant and engines — essentially its 'maneuvering budget.' In the Mission Simulation, your <strong>Engine Propulsion</strong> percentage represents how much of your total budget is devoted to engines, propellant capacity, and propulsion-related hardware, which in turn determines your effective Delta-v capability for the chosen destination.</p><p><strong>Why this matters per-destination:</strong> A mission to Mars orbit requires substantially more Delta-v than a lunar mission, and a mission to the outer asteroid belt or an icy moon like Europa requires more still. If your propulsion allocation is too low for an ambitious destination, your spacecraft may simply lack the maneuvering capability to reach it, enter orbit correctly, or make necessary course corrections — regardless of how well-protected or well-equipped it otherwise is.</p><p><strong>Engineering intuition:</strong> Propulsion budget is not just about 'getting there' — it also covers mid-course correction burns, orbital insertion burns, and any planned maneuvers after arrival. Under-budgeting propulsion is one of the most common ways an otherwise well-designed mission fails, because there is generally no way to add more propellant once the spacecraft has launched.</p><p><strong>Common misconception:</strong> Assuming propulsion budget only matters for the initial launch. In reality, arrival maneuvers (like orbital insertion) often require just as much precision and Delta-v margin as the outbound journey.</p>",
        "fact": "Many interplanetary missions carry a dedicated Delta-v reserve specifically for orbital insertion — arriving at the destination is often just as propulsion-intensive as the journey there.",
        "misconception": "Propulsion budget is not just about launch and cruise. Arrival maneuvers like orbital insertion can require comparable Delta-v, so under-budgeting propulsion risks failure even after a successful journey.",
        "application": "In the simulator, weigh your Engine Propulsion slider against your chosen destination's likely Delta-v demands — deep-space or high-orbit destinations generally warrant a higher propulsion allocation.",
        "summary": "Engine Propulsion budget determines a mission's Delta-v capability, which must cover both the outbound journey and arrival maneuvers — making it a critical allocation that varies significantly by destination.",
        "visualType": "anatomy"
    },
    {
        "id": 6,
        "title": "Module 6: Protection, Shielding & Space Weather Risk",
        "subtitle": "Why 'Protection & Shielding %' Interacts Directly With Live Space Weather",
        "badge": "SHIELDING",
        "intro": "Your Protection & Shielding budget determines how well your spacecraft can withstand radiation, thermal extremes, and micrometeorite impacts — and the right amount depends on both destination and current space weather.",
        "lesson": "<p>Recall from Rocket Academy I, Module 10, that solar flares and coronal mass ejections pose genuine risks to spacecraft electronics and crew safety. In the Mission Simulation, your <strong>Protection & Shielding</strong> percentage represents the portion of your budget dedicated to radiation shielding, thermal protection, and structural hardening against micrometeorite impacts.</p><p><strong>Why this connects to the live NASA weather feed:</strong> The simulator pulls real, current space weather data from NASA's DONKI system. If there's active solar flare or CME activity at the time you run your evaluation, a protection allocation that would otherwise be adequate may be flagged as insufficient — exactly as a real mission planner would need to reassess shielding requirements ahead of a launch window during elevated solar activity.</p><p><strong>Engineering intuition:</strong> Protection budget isn't just about electronics — for crewed missions, or even sensitive robotic payloads, inadequate shielding can compromise the entire mission regardless of how capable the payload or how efficient the propulsion. Just as importantly (recall Rocket Academy I's misconception on this point), protection is not simply 'the more the better' — over-allocating protection budget starves propulsion and payload without necessarily addressing the specific risk profile of the chosen destination.</p><p><strong>Common misconception:</strong> Assuming protection budget is a fixed, destination-independent requirement. In reality, a deep-space mission facing months of interplanetary radiation exposure warrants meaningfully more protection budget than a short lunar mission.</p>",
        "fact": "This platform's live NASA DONKI weather feed is not decorative — the AI evaluation engine factors current solar flare and CME activity directly into whether your protection allocation is judged sufficient.",
        "misconception": "Protection budget is not a fixed, one-size-fits-all requirement. The right amount depends on destination, mission duration, and current space weather conditions — not a single universal target.",
        "application": "Before running your Mission Simulation, check the live space weather feed — if a flare is active, consider whether your Protection & Shielding allocation is adequate for the conditions.",
        "summary": "Protection & Shielding budget determines resilience against radiation, thermal extremes, and impacts, and the appropriate allocation depends on destination and real-time space weather, not a fixed universal amount.",
        "visualType": "weather"
    },
    {
        "id": 7,
        "title": "Module 7: Mass Constraints & Structural Margins",
        "subtitle": "Why You Can't Simply Add More of Everything",
        "badge": "MARGINS",
        "intro": "Every spacecraft has a hard mass ceiling set by its launch vehicle's capability — and engineers must reserve structural margin within that ceiling to handle real-world uncertainty.",
        "lesson": "<p>A launch vehicle can only lift a certain maximum mass to a given orbit or trajectory — this is a hard physical ceiling, not a soft target. Everything the mission needs — propellant, structure, shielding, payload — must fit within that ceiling. This is why your three budget sliders in the Mission Simulation aren't independent free choices; they compete for the same fixed resource, mirroring how a real mission's mass budget is a zero-sum allocation.</p><p><strong>Structural margin</strong> is the deliberate practice of not using 100% of your theoretical mass or budget capacity, reserving a buffer for unexpected requirements discovered later in development, manufacturing tolerances, or in-flight anomalies. A mission designed with zero margin is fragile — the smallest surprise (a slightly heavier-than-expected instrument, a software patch requiring more memory hardware) can force painful late-stage redesigns.</p><p><strong>Engineering intuition:</strong> Experienced planners deliberately avoid allocating budget right up to 100% of any single category if it can be avoided, because doing so leaves zero flexibility to respond to problems discovered during the AI evaluation or later mission phases. A margin-conscious plan trades a small amount of theoretical maximum performance for a much larger reduction in overall mission risk.</p><p><strong>Common misconception:</strong> Assuming that using every available percentage point of budget in your favorite category is always the most efficient choice. In real engineering, reserved margin is not wasted resource — it's insurance against failure.</p>",
        "fact": "Aerospace programs often reserve 10-30% mass margin early in development specifically because historical data shows spacecraft mass reliably grows as a design matures — a phenomenon informally known as 'mass creep.'",
        "misconception": "Using every available percentage point of your budget in one category is not automatically the most efficient plan. Reserved margin protects against 'mass creep' and unexpected requirements discovered later.",
        "application": "When configuring your sliders in the simulator, consider leaving some conceptual margin rather than pushing any single allocation to its absolute maximum — this mirrors real engineering discipline.",
        "summary": "Mass constraints set a hard ceiling on total spacecraft capability, and deliberate structural margin — not using every last percentage point — protects a mission against real-world uncertainty and 'mass creep.'",
        "visualType": "tradeoff"
    },
    {
        "id": 8,
        "title": "Module 8: Reliability & Safety Margins",
        "subtitle": "Designing for the Failures You Cannot Predict",
        "badge": "RELIABILITY",
        "intro": "Reliability engineering is the discipline of ensuring a mission still succeeds even when individual components fail, using redundancy and safety margins rather than assuming perfect performance.",
        "lesson": "<p><strong>Reliability</strong> is the probability that a system performs its intended function without failure over its mission lifetime. No component — engine, sensor, computer, or structural joint — has a zero failure rate, so reliability engineering assumes failures will happen and designs the overall system to tolerate them anyway.</p><p>The primary tool for this is <strong>redundancy</strong>: carrying backup systems (a second computer, a second communications radio, a second set of thrusters) so that a single component failure doesn't end the mission. Redundancy adds mass and cost — connecting directly back to the trade-offs of Modules 4 and 7 — so engineers apply it selectively to the most mission-critical systems rather than everywhere uniformly.</p><p><strong>Safety margin</strong> is closely related but distinct: rather than duplicating a system, engineers design individual components to withstand more stress than they are ever expected to encounter in normal operation, providing a buffer against manufacturing variability, unexpected environmental extremes, or measurement uncertainty.</p><p><strong>Engineering intuition:</strong> In the Mission Simulation, a spacecraft configuration with a high readiness rating generally reflects sensible balance and adequate margin across categories, not maximum allocation in any single category. A mission overloaded with payload and thin on protection or propulsion will typically be evaluated as high-risk, because there's little margin to absorb any variance from ideal conditions.</p><p><strong>Common misconception:</strong> Assuming reliability is achieved simply by using higher-quality individual components. In practice, reliability comes primarily from system-level design choices — redundancy and margin — not from any single component being flawless.</p>",
        "fact": "Many robotic space missions carry two or three redundant computers running in parallel specifically so that a single radiation-induced computer fault (recall the Single Event Upsets from Rocket Academy I) cannot end the mission.",
        "misconception": "Reliability isn't achieved primarily through flawless individual components. It comes from system-level design choices — redundancy and margin — that tolerate the failures that will inevitably occur.",
        "application": "The readiness rating you receive from the AI evaluation reflects exactly this kind of system-level reliability thinking, rewarding balanced, margin-conscious configurations over extreme, unbalanced ones.",
        "summary": "Reliability engineering assumes components will fail and uses redundancy and safety margin to keep the overall mission succeeding anyway — a mindset directly reflected in how the simulator evaluates your mission's readiness.",
        "visualType": "team"
    },
    {
        "id": 9,
        "title": "Module 9: Mission Risk Assessment & Feasibility",
        "subtitle": "Judging Whether a Configuration Can Realistically Succeed",
        "badge": "RISK",
        "intro": "Risk assessment is the structured process of identifying what could go wrong, how likely it is, and how severe the consequences would be — forming the basis of every GO/NO-GO decision.",
        "lesson": "<p>Before any real mission launches, engineers conduct formal risk assessments: cataloguing potential failure modes (a propulsion shortfall, inadequate shielding against an active solar storm, structural failure under an unexpected load), estimating how likely each is, and evaluating how severe the consequences would be if it occurred. This process feeds directly into a mission's overall <strong>feasibility</strong> — whether the configuration, as designed, can realistically achieve its objective.</p><p><strong>How this maps to the simulator:</strong> When you run your Mission Simulation, the AI evaluation engine is effectively performing a compressed version of this risk assessment — checking whether your propulsion allocation is plausible for your chosen destination, whether your protection allocation is adequate given current space weather, and whether your payload allocation has left enough margin elsewhere. The resulting <strong>readiness rating</strong> and <strong>GO / NO-GO launch status</strong> are direct analogues of a real mission review board's verdict.</p><p><strong>Engineering intuition:</strong> A 'NO-GO' evaluation is not a failure of the exercise — it's exactly the outcome a real risk assessment is designed to catch before a mission commits irreversible resources. Iterating on your configuration in response to a NO-GO evaluation mirrors exactly how real engineering teams respond to a failed risk review: adjust the design, not the destination's requirements.</p><p><strong>Common misconception:</strong> Treating a NO-GO result as an error in the simulator rather than useful engineering feedback. The entire purpose of risk assessment is to surface exactly this kind of warning before committing to a real, irreversible launch.</p>",
        "fact": "Real spacecraft missions go through multiple formal 'Flight Readiness Reviews' before launch, where independent engineering panels can still issue a NO-GO recommendation even very close to a scheduled launch date.",
        "misconception": "A NO-GO evaluation in the simulator is not a bug or a failure of the exercise — it reflects exactly the kind of risk-assessment finding that prevents real missions from launching with an inadequate configuration.",
        "application": "Use a NO-GO or low readiness rating the same way a real mission team would: as a prompt to rebalance your propulsion, protection, and payload allocations, not as an error to dismiss.",
        "summary": "Risk assessment identifies potential failure modes and their consequences, directly determining a mission's feasibility — and the simulator's readiness rating and GO/NO-GO status are a compressed version of this real engineering process.",
        "visualType": "tradeoff"
    },
    {
        "id": 10,
        "title": "Module 10: Launch Windows & Readiness Evaluation",
        "subtitle": "Bringing Every Concept Together for the Capstone Simulation",
        "badge": "READINESS",
        "intro": "A launch window is the limited period during which orbital mechanics and environmental conditions align favorably for a mission to depart — and readiness evaluation confirms the mission is actually prepared to use it.",
        "lesson": "<p>Because destinations move (planets orbit the Sun, the Moon orbits Earth), missions can't launch at just any moment — they must depart during a <strong>launch window</strong>, a period when the relative positions of Earth and the destination allow an efficient trajectory using a reasonable Delta-v budget. Missing a launch window can mean waiting weeks, months, or in some interplanetary cases, years for the next favorable alignment.</p><p>Space weather (Module 6) can further compress a launch window — an active solar storm might make an otherwise perfect orbital alignment too risky to fly through. This is exactly why the Mission Simulation integrates live NASA space weather data alongside your budget configuration: real readiness evaluation always combines orbital timing with environmental conditions, not just spacecraft design in isolation.</p><p><strong>Bringing it all together:</strong> By this point you've learned why destination drives requirements (Module 2), why payload competes with propulsion and protection (Modules 3-4), why propulsion budget must match Delta-v needs (Module 5), why protection must respond to both destination and live space weather (Module 6), why margin and redundancy matter (Modules 7-8), and how risk assessment produces a GO/NO-GO verdict (Module 9). The Mission Simulation is designed to exercise every one of these concepts in a single integrated decision, exactly as a real mission planning team would.</p><p><strong>Common misconception:</strong> Believing that a 'readiness rating' is an arbitrary score. In reality it synthesizes exactly the trade-offs and risk factors you've studied throughout this academy — engine allocation relative to destination demands, protection relative to current conditions, and overall balance and margin.</p>",
        "fact": "Interplanetary launch windows to Mars occur roughly every 26 months, when Earth and Mars align favorably — missing one can mean waiting over two years for the next opportunity.",
        "misconception": "The readiness rating in the simulator is not an arbitrary number. It synthesizes the same trade-offs — destination demands, protection adequacy, propulsion sufficiency, and overall balance — that a real mission readiness review would evaluate.",
        "application": "You are now ready for the Mission Simulation capstone: configure your destination, budget, and allocations with everything you've learned in both academies in mind, then review the AI evaluation as genuine engineering feedback.",
        "summary": "Launch windows combine orbital mechanics with environmental conditions to define when a mission can depart, and the Mission Simulation's readiness evaluation integrates every concept from this academy into one capstone engineering decision.",
        "visualType": "launch"
    }
]

ACADEMY_2_QUESTIONS = {
    1: [
        {
            "question": "Why should mission objectives be defined before any engineering design work begins?",
            "options": ["Because it's a legal requirement", "Because the objective determines which engineering priorities and trade-offs actually matter for this mission", "Because objectives are unrelated to engineering decisions", "Because budgets are fixed regardless of objective"],
            "correct": 1,
            "reasoning": "A mission's objective defines what success means, which in turn determines which engineering trade-offs are worth prioritizing."
        },
        {
            "question": "What does 'systems thinking' mean in the context of mission planning?",
            "options": ["Focusing only on the propulsion subsystem", "Recognizing that a spacecraft's subsystems are interconnected, so a change in one affects the others", "Using more computers on the spacecraft", "Ignoring budget constraints in favor of best-case design"],
            "correct": 1,
            "reasoning": "Systems thinking means recognizing that decisions in one subsystem (like adding shielding mass) ripple into requirements for other subsystems (like propulsion)."
        },
        {
            "question": "What is a common misconception about mission configuration that systems thinking corrects?",
            "options": ["That trade-offs exist at all", "That there is a single configuration that maximizes every metric simultaneously", "That objectives matter", "That budgets are unlimited"],
            "correct": 1,
            "reasoning": "In real engineering, nearly every decision trades one advantage against another — there is rarely a configuration that maximizes everything at once."
        },
        {
            "question": "How does the Mission Simulation reflect systems thinking in practice?",
            "options": ["By requiring you to allocate a fixed budget across competing priorities that affect each other", "By allowing unlimited budget for every category", "By ignoring destination entirely", "By only evaluating propulsion"],
            "correct": 0,
            "reasoning": "The simulator requires balancing budget across propulsion, protection, and payload, mirroring how real systems-level trade-offs work."
        },
        {
            "question": "Why did NASA's James Webb Space Telescope require over two decades of trade-off work?",
            "options": ["Because of unrelated funding delays only", "Because balancing its mirror size against launch vehicle fairing constraints required extensive systems-level trade studies", "Because the telescope had no defined objective", "Because it used no shielding"],
            "correct": 1,
            "reasoning": "Fitting a large mirror within launch vehicle size constraints required extensive systems engineering trade-off analysis over many years."
        }
    ],
    2: [
        {
            "question": "Why does destination selection affect nearly every other mission parameter?",
            "options": ["It doesn't — destination is a purely cosmetic choice", "Because it determines Delta-v requirements, travel time, and environmental hazards the mission must plan around", "Because destinations all require identical propulsion budgets", "Because destination only affects mission naming"],
            "correct": 1,
            "reasoning": "Destination determines the Delta-v needed, the trip duration, and the specific hazards (radiation, thermal, dust) the mission will encounter."
        },
        {
            "question": "Why does a Mars mission generally require more protection consideration than a short lunar mission?",
            "options": ["Mars missions require no protection at all", "A Mars mission spends months in interplanetary space exposed to solar radiation before arrival", "Lunar missions face more radiation than Mars missions", "Protection needs are identical regardless of destination"],
            "correct": 1,
            "reasoning": "Longer interplanetary transit times mean more cumulative radiation exposure, generally warranting greater protection allocation than a short lunar trip."
        },
        {
            "question": "What is a common misconception about choosing a mission destination in the simulator?",
            "options": ["That destination affects Delta-v needs", "That destination is a purely cosmetic choice unrelated to budget allocation", "That destination affects protection needs", "That destination affects trip duration"],
            "correct": 1,
            "reasoning": "Destination is not cosmetic — it is the foundational decision from which propulsion, protection, and other budget needs are derived."
        },
        {
            "question": "Approximately how long does a one-way trip to Mars typically take using efficient transfer trajectories?",
            "options": ["A few hours", "A few days", "Six to nine months", "Several decades"],
            "correct": 2,
            "reasoning": "Efficient Mars transfer trajectories typically take six to nine months one-way, which is why radiation exposure is a much bigger concern than for a lunar mission."
        },
        {
            "question": "Which of these destination-specific hazards is most associated with an icy moon like Europa?",
            "options": ["Intense radiation belts from the host planet, such as Jupiter", "Abrasive lunar regolith dust", "Asteroid belt navigation hazards only", "No notable environmental hazards"],
            "correct": 0,
            "reasoning": "Icy moons orbiting giant planets like Jupiter are subject to intense radiation belts generated by the planet's magnetic field, a major consideration for mission protection budgets."
        }
    ],
    3: [
        {
            "question": "Why isn't 'maximize payload allocation' always the best strategy in mission planning?",
            "options": ["Because payload never affects mission value", "Because payload mass draws from the same limited budget as propulsion and protection, and an under-protected or underpowered spacecraft may never deliver that payload's value", "Because payload is always the cheapest category", "Because payload allocation is fixed by regulation"],
            "correct": 1,
            "reasoning": "Overbuilding payload at the expense of propulsion or protection can leave a mission unable to reach or survive its destination, delivering zero value from an otherwise capable payload."
        },
        {
            "question": "What question should mission planners ask when setting payload allocation, according to this module?",
            "options": ["How much payload can we possibly fit?", "What is the minimum payload capability that still achieves the mission objective?", "Should we ignore payload entirely?", "Can payload replace propulsion?"],
            "correct": 1,
            "reasoning": "Experienced planners size payload to the mission's actual objective rather than maximizing it, preserving budget for propulsion and protection."
        },
        {
            "question": "What does the James Webb Space Telescope's tennis-court-sized sunshield illustrate about payload planning?",
            "options": ["That payload and protection needs are often deeply intertwined", "That sunshields are unrelated to payload function", "That all payloads require sunshields", "That payload mass is irrelevant to mission design"],
            "correct": 0,
            "reasoning": "The sunshield exists specifically to keep the payload (mirror and instruments) cold enough to function, showing how protection and payload requirements can be interdependent."
        },
        {
            "question": "In the Mission Simulation, what does the 'Science Payloads' percentage represent?",
            "options": ["The portion of budget dedicated to payload capability, competing directly with propulsion and protection budgets", "The total mission budget in dollars", "The destination selected", "The current space weather status"],
            "correct": 0,
            "reasoning": "The Science Payloads slider represents how much of the fixed total budget is allocated to payload, directly trading off against propulsion and protection."
        },
        {
            "question": "What is a common misconception this module addresses about payload budget?",
            "options": ["That payload is the reason a mission exists", "That maximizing payload allocation always maximizes mission value", "That payload competes with other budgets", "That payload includes scientific instruments"],
            "correct": 1,
            "reasoning": "Maximizing payload budget does not automatically maximize value — a spacecraft that never survives to use its payload delivers no value regardless of payload capability."
        }
    ],
    4: [
        {
            "question": "Why are engineering trade-offs unavoidable when a mission has a fixed total budget?",
            "options": ["Because propulsion, protection, and payload compete directly for the same limited resource", "Because trade-offs are optional in real engineering", "Because budgets are never actually fixed", "Because only one category matters"],
            "correct": 0,
            "reasoning": "With a fixed budget, allocating more to one category (like payload) necessarily leaves less available for the others (propulsion, protection)."
        },
        {
            "question": "What is a common misconception about the 'ideal' way to split budget across propulsion, protection, and payload?",
            "options": ["That trade-offs exist at all", "That an even, balanced 33/33/33 split is always optimal regardless of destination and objective", "That budget is limited", "That destination matters"],
            "correct": 1,
            "reasoning": "The optimal split depends on destination and mission objective — an even split is not universally correct, and sometimes an intentionally unbalanced allocation is the right engineering choice."
        },
        {
            "question": "What question do experienced mission planners ask instead of 'what's the best possible allocation'?",
            "options": ["What allocation maximizes every metric at once?", "What does this specific destination, on this specific budget, actually require to succeed?", "How can we ignore the budget constraint?", "Which slider should be set to zero?"],
            "correct": 1,
            "reasoning": "Rather than seeking a universally 'best' allocation, planners tailor the balance to the specific mission's destination and requirements."
        },
        {
            "question": "Which three categories directly compete for a mission's fixed budget in the simulator?",
            "options": ["Engine Propulsion, Protection & Shielding, and Science Payloads", "Destination, weather, and readiness rating", "Launch date, crew size, and mission name", "Communications, navigation, and marketing"],
            "correct": 0,
            "reasoning": "The three sliders — Engine Propulsion, Protection & Shielding, and Science Payloads — represent the core trade-off categories that must sum within a fixed total budget."
        },
        {
            "question": "Why did NASA's Perseverance rover mission spend years on trade studies balancing sampling systems, power systems, and scientific instruments?",
            "options": ["Because these systems compete for the same limited mass and budget resources", "Because NASA had unlimited budget for the mission", "Because trade-offs don't apply to rovers", "Because sampling systems require no power"],
            "correct": 0,
            "reasoning": "Like any mission, Perseverance's subsystems competed for limited mass and budget, requiring extensive trade-off analysis to finalize a balanced design."
        }
    ],
    5: [
        {
            "question": "What does the Engine Propulsion budget in the simulator ultimately determine?",
            "options": ["The mission's Delta-v capability for the chosen destination", "The current space weather conditions", "The mission's payload capability", "The launch vehicle's paint color"],
            "correct": 0,
            "reasoning": "Engine Propulsion budget determines how much Delta-v — maneuvering capability — the spacecraft has available for its journey and arrival."
        },
        {
            "question": "Why might a mission to the asteroid belt or an icy moon like Europa require a higher propulsion allocation than a lunar mission?",
            "options": ["Because deeper, more distant destinations generally demand more Delta-v", "Because farther destinations always require less propulsion", "Because propulsion needs are identical for all destinations", "Because icy moons have no gravity"],
            "correct": 0,
            "reasoning": "More distant or complex destinations typically require significantly more Delta-v, and therefore a larger propulsion budget, than a relatively nearby lunar mission."
        },
        {
            "question": "What is a common misconception about when propulsion budget matters most?",
            "options": ["That it only matters for the initial launch, not arrival maneuvers", "That it affects Delta-v", "That it varies by destination", "That it competes with other budgets"],
            "correct": 0,
            "reasoning": "Propulsion budget matters throughout the mission — arrival maneuvers like orbital insertion can require just as much Delta-v as the outbound journey."
        },
        {
            "question": "Why do many interplanetary missions carry a dedicated Delta-v reserve for orbital insertion?",
            "options": ["Because arriving at a destination can require comparable propulsive effort to reaching it", "Because orbital insertion requires no propulsion at all", "Because reserves are only used during launch", "Because insertion burns are optional"],
            "correct": 0,
            "reasoning": "Orbital insertion at the destination often demands significant Delta-v, so missions plan for it just as carefully as the outbound journey."
        },
        {
            "question": "What risk does under-budgeting propulsion create, given that propellant generally cannot be added after launch?",
            "options": ["No risk — propulsion can always be resupplied in flight", "The spacecraft may lack the maneuvering capability to reach or properly enter orbit at its destination", "It only affects payload capability", "It only affects protection needs"],
            "correct": 1,
            "reasoning": "Since propellant typically cannot be replenished after launch, an under-budgeted propulsion allocation risks leaving the spacecraft unable to complete necessary maneuvers."
        }
    ],
    6: [
        {
            "question": "What does the Protection & Shielding budget in the simulator primarily guard against?",
            "options": ["Radiation, thermal extremes, and micrometeorite impacts", "Software bugs only", "Launch vehicle staging failures", "Payload budget overruns"],
            "correct": 0,
            "reasoning": "Protection and shielding budget covers radiation shielding, thermal protection, and structural hardening against impacts."
        },
        {
            "question": "How does the simulator's live NASA weather feed connect to your Protection & Shielding allocation?",
            "options": ["It has no connection to protection allocation", "Active solar flare or CME activity can make an otherwise adequate protection allocation be flagged as insufficient", "It only affects the destination dropdown", "It replaces the need for shielding entirely"],
            "correct": 1,
            "reasoning": "The AI evaluation factors current space weather into whether your protection budget is judged adequate, just as real mission planners reassess shielding ahead of active solar events."
        },
        {
            "question": "What is a common misconception about protection budget addressed in this module?",
            "options": ["That it responds to space weather", "That it is a fixed, destination-independent requirement rather than something that should scale with mission duration and destination", "That it involves radiation shielding", "That it can be evaluated by the AI engine"],
            "correct": 1,
            "reasoning": "Protection needs vary by destination and mission duration — a short lunar trip and a months-long deep-space journey do not warrant identical protection allocations."
        },
        {
            "question": "Why is protection budget not simply 'more is always better'?",
            "options": ["Because over-allocating protection starves propulsion and payload without necessarily matching the mission's actual risk profile", "Because shielding materials are free", "Because protection budget has no upper limit", "Because protection never affects other budgets"],
            "correct": 0,
            "reasoning": "Since budget is fixed, over-investing in protection reduces what's available for propulsion and payload, so allocation should match actual risk rather than be maximized blindly."
        },
        {
            "question": "What should you check before running your Mission Simulation evaluation, according to this module?",
            "options": ["The live space weather feed, to judge whether your protection allocation is adequate for current conditions", "The mission's launch date only", "The number of crew members", "The color scheme of the interface"],
            "correct": 0,
            "reasoning": "Reviewing live space weather before evaluating your configuration lets you judge whether your protection budget is adequate for current solar activity, just as real mission planners would."
        }
    ],
    7: [
        {
            "question": "Why can't a mission simply add more of everything — more propulsion, more protection, more payload?",
            "options": ["Because a launch vehicle has a hard mass ceiling, and a fixed budget forces trade-offs among categories", "Because there is no limit to what a launch vehicle can lift", "Because propulsion, protection, and payload are unrelated to mass", "Because budgets are unlimited in real missions"],
            "correct": 0,
            "reasoning": "Launch vehicles have a fixed maximum liftable mass, and a fixed budget means every category competes for the same limited resource."
        },
        {
            "question": "What is 'structural margin' in mission planning?",
            "options": ["Using every available percentage point of budget to maximize performance", "Deliberately reserving a buffer within mass or budget capacity to handle uncertainty and unexpected requirements", "A type of rocket engine", "A measurement of orbital altitude"],
            "correct": 1,
            "reasoning": "Structural margin is a deliberate reserve left unused to absorb unexpected requirements discovered later in development or during flight."
        },
        {
            "question": "What is 'mass creep,' as referenced in this module?",
            "options": ["The tendency for spacecraft mass to grow reliably as a design matures", "A type of propulsion failure", "A navigation error near asteroids", "A software bug in guidance systems"],
            "correct": 0,
            "reasoning": "Mass creep is the well-documented tendency for spacecraft designs to gain mass as development proceeds, which is why engineers reserve margin in advance."
        },
        {
            "question": "What is a common misconception about using 100% of a budget category in the simulator?",
            "options": ["That it always represents the most efficient, risk-free plan", "That budgets are fixed", "That margin exists in real engineering", "That mass constraints are real"],
            "correct": 0,
            "reasoning": "Using every available percentage point in one category is not necessarily most efficient — reserved margin protects against unforeseen problems rather than being wasted resource."
        },
        {
            "question": "Roughly what percentage of mass margin do aerospace programs often reserve early in development?",
            "options": ["0%, since margin is unnecessary", "10-30%", "90-100%", "Exactly 50%, by regulation"],
            "correct": 1,
            "reasoning": "Aerospace programs commonly reserve roughly 10-30% mass margin early on, anticipating that mass will grow as the design matures."
        }
    ],
    8: [
        {
            "question": "What is 'redundancy' in reliability engineering?",
            "options": ["Removing backup systems to save mass", "Carrying backup systems, like a second computer or radio, so a single component failure doesn't end the mission", "A measurement of propulsion efficiency", "A type of space weather event"],
            "correct": 1,
            "reasoning": "Redundancy means carrying duplicate critical systems so that one failure does not cause total mission loss."
        },
        {
            "question": "How is 'safety margin' distinct from redundancy?",
            "options": ["They are the same concept with different names", "Safety margin designs individual components to withstand more stress than expected, rather than duplicating entire systems", "Safety margin only applies to crewed missions", "Safety margin eliminates the need for redundancy entirely"],
            "correct": 1,
            "reasoning": "Safety margin over-engineers individual components against uncertainty, while redundancy duplicates whole systems — they are related but distinct reliability strategies."
        },
        {
            "question": "What is a common misconception about how reliability is achieved?",
            "options": ["That it comes from system-level design choices like redundancy and margin", "That it is achieved primarily by using flawless individual components rather than system-level design", "That failures are assumed to happen", "That redundancy adds mass"],
            "correct": 1,
            "reasoning": "Reliability comes mainly from system-level strategies like redundancy and margin, not from assuming any single component will never fail."
        },
        {
            "question": "Why do many robotic missions carry two or three redundant flight computers?",
            "options": ["So a single radiation-induced fault, like a Single Event Upset, cannot end the mission", "Because computers are the lightest components on a spacecraft", "Because redundancy is required for aesthetic reasons", "Because a single computer cannot run mission software"],
            "correct": 0,
            "reasoning": "Redundant computers protect against radiation-induced faults (Single Event Upsets) that could otherwise disable a single computer and jeopardize the mission."
        },
        {
            "question": "What does a high readiness rating in the Mission Simulation generally reflect?",
            "options": ["Maximum allocation in a single budget category", "Sensible balance and adequate margin across categories, rather than an extreme configuration", "The lowest possible total budget", "A destination choice alone"],
            "correct": 1,
            "reasoning": "High readiness ratings generally reward balanced, margin-conscious configurations rather than extreme allocations concentrated in one category."
        }
    ],
    9: [
        {
            "question": "What is the purpose of a formal mission risk assessment?",
            "options": ["To identify potential failure modes, estimate their likelihood, and evaluate the severity of their consequences", "To guarantee a mission will never fail", "To eliminate the need for a launch window", "To replace the need for a budget"],
            "correct": 0,
            "reasoning": "Risk assessment systematically catalogues what could go wrong, how likely it is, and how severe the consequences would be, informing the overall feasibility judgment."
        },
        {
            "question": "How does the Mission Simulation's AI evaluation engine relate to real mission risk assessment?",
            "options": ["It performs a compressed version of this process, checking propulsion, protection, and payload plausibility against destination and space weather", "It has no relationship to real risk assessment", "It only checks the destination field", "It ignores space weather entirely"],
            "correct": 0,
            "reasoning": "The simulator's evaluation mirrors real risk assessment by checking whether allocations are plausible given destination requirements and current conditions."
        },
        {
            "question": "What is a common misconception about receiving a 'NO-GO' result in the simulator?",
            "options": ["That it reflects useful engineering feedback consistent with real risk assessment", "That it is a bug or error in the exercise rather than meaningful feedback", "That it can be resolved by adjusting the budget", "That it relates to protection allocation"],
            "correct": 1,
            "reasoning": "A NO-GO is not a malfunction — it reflects the same kind of finding a real risk assessment is designed to surface before a mission commits irreversible resources."
        },
        {
            "question": "What should a mission planner do in response to a NO-GO evaluation, according to this module?",
            "options": ["Ignore it and launch anyway", "Adjust the design — such as rebalancing budget allocations — rather than changing the destination's inherent requirements", "Assume the evaluation engine is broken", "Immediately abandon the mission concept entirely"],
            "correct": 1,
            "reasoning": "A NO-GO calls for redesign of the configuration to better meet the destination's real requirements, just as real engineering teams respond to a failed risk review."
        },
        {
            "question": "What is a Flight Readiness Review, as referenced in this module?",
            "options": ["A formal review process where independent engineering panels can still issue a NO-GO recommendation close to a scheduled launch", "An informal chat between astronauts before launch", "A post-launch celebration event", "A software update applied after landing"],
            "correct": 0,
            "reasoning": "Flight Readiness Reviews are formal engineering evaluations that can still block a launch with a NO-GO recommendation, even close to the scheduled date."
        }
    ],
    10: [
        {
            "question": "What is a 'launch window'?",
            "options": ["The physical opening on a launch pad where a rocket sits", "A limited period when the relative positions of Earth and the destination allow an efficient trajectory", "The time it takes to fuel a rocket", "A window on the spacecraft for viewing Earth"],
            "correct": 1,
            "reasoning": "A launch window is the limited period during which orbital alignment allows an efficient trajectory to the destination using a reasonable Delta-v budget."
        },
        {
            "question": "How can space weather further affect a launch window?",
            "options": ["It has no effect on launch windows", "An active solar storm might make an otherwise favorable orbital alignment too risky to fly through", "Space weather only matters after landing", "Space weather extends every launch window automatically"],
            "correct": 1,
            "reasoning": "Even during a favorable orbital alignment, elevated solar activity can add enough risk that mission planners choose to delay, compressing the effective launch window."
        },
        {
            "question": "Roughly how often do favorable launch windows to Mars occur?",
            "options": ["Every few days", "Roughly every 26 months", "Only once per century", "Continuously, with no meaningful gap"],
            "correct": 1,
            "reasoning": "Mars launch windows occur roughly every 26 months when Earth and Mars align favorably; missing one can mean a wait of over two years."
        },
        {
            "question": "What is a common misconception about the Mission Simulation's readiness rating?",
            "options": ["That it is an arbitrary number unrelated to the concepts taught in this academy", "That it reflects destination demands and protection adequacy", "That it reflects propulsion sufficiency", "That it reflects overall balance and margin"],
            "correct": 0,
            "reasoning": "The readiness rating is not arbitrary — it synthesizes destination demands, protection adequacy, propulsion sufficiency, and overall balance, exactly as covered throughout this academy."
        },
        {
            "question": "What is the intended relationship between Rocket Academy II and the Mission Simulation, according to this module?",
            "options": ["They are unrelated activities", "The Mission Simulation is a capstone that exercises every concept taught across this academy in one integrated decision", "The simulation replaces the need for Rocket Academy II", "The simulation only tests destination selection"],
            "correct": 1,
            "reasoning": "The Mission Simulation is explicitly designed as a capstone experience that applies every concept from Rocket Academy II — objectives, destination, payload, budget, propulsion, protection, margin, reliability, and risk assessment — together."
        }
    ]
}


def get_randomized_questions(academy_id, module_id):
    """Retrieves question sets for a given academy + module for dynamic student assessment."""
    bank = ACADEMY_1_QUESTIONS if str(academy_id) == '1' else ACADEMY_2_QUESTIONS
    q_list = bank.get(module_id, bank[1])
    processed = []
    for q in q_list:
        q_copy = dict(q)
        processed.append(q_copy)
    return processed


@app.route('/api/academy', methods=['GET'])
def get_academy_content():
    """
    Serves structured educational modules for either Rocket Academy I
    (Aerospace Foundations) or Rocket Academy II (Mission Planning Academy),
    selected via ?academy=1 or ?academy=2 (defaults to 1).
    """
    academy_id = request.args.get('academy', '1')

    if str(academy_id) == '2':
        source_modules = ACADEMY_2_MODULES
        academy_title = "Rocket Academy II: Mission Planning Academy"
        academy_subtitle = "Preparing you for every decision inside the Mission Simulation"
    else:
        academy_id = '1'
        source_modules = ACADEMY_1_MODULES
        academy_title = "Rocket Academy I: Aerospace Foundations"
        academy_subtitle = "Undergraduate Aerospace Engineering Coursework — start here with zero prior knowledge"

    modules = [dict(m) for m in source_modules]
    for mod in modules:
        mod["questions"] = get_randomized_questions(academy_id, mod["id"])

    return jsonify({
        "modules": modules,
        "academy": academy_id,
        "academy_title": academy_title,
        "academy_subtitle": academy_subtitle,
        "status": "OK"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
