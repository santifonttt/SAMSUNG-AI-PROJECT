
# To run this app, use the command:
# streamlit run app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import tempfile
import tensorflow_hub as hub
import json
import joblib
from groq import Groq # Change if you use OpenAI instead

st.set_page_config(page_title="AI Mechanic Assistant", page_icon="🚗")

# ------------------------------------------------------------
# LLM PROMPT
# ------------------------------------------------------------
prompt = """
You are an AI automotive mechanic assistant. You MUST output ONLY valid JSON with no Markdown, no backticks, and no extra text.

You will receive a small set of predicted problems (top-1 or top-2). You must only return JSON objects for the problems provided.

STRICT OUTPUT RULES:
1. If only top-1 is provided, output EXACTLY one JSON object inside a JSON array.
2. If top-2 is provided, output EXACTLY two JSON objects.
3. NEVER output more problems than those provided.

PROBLEM LABELS (allowed values):
- "Worn Out Brakes"
- "Normal Brakes"
- "Serpentine Belt"
- "Car Knocking"
- "Normal Engine Idle"
- "Power Steering"
- "Low Oil"
- "Bad Ignition"
- "Dead Battery"
- "Normal Engine Sound"
- "Normal Engine Startup"

SPECIAL RULE FOR "NORMAL" PROBLEMS:
If the problem contains "Normal":
- Explain that the sound/behavior is normal.
- Provide light monitoring advice.

CRITICAL GLOBAL RULE (APPLIES TO EVERY FIELD OF THE JSON):
You MUST NOT mention, imply, or describe any cause that corresponds to ANY item from the official problem list.

FORBIDDEN TERMS (DO NOT APPEAR ANYWHERE IN THE OUTPUT):
- anything equivalent to any problem label

Make clear explanations.

OUTPUT FORMAT FOR EACH PROBLEM:
{
  "problem": <string>,
  "probability": <float>,
  "severity": <"low" | "moderate" | "high">,
  "explanation": <string WITHOUT any forbidden terms>,
  "possible_causes": <list WITHOUT any forbidden terms>,
  "recommended_actions": <list WITHOUT any forbidden terms>,
  "what_to_tell_the_mechanic": <string WITHOUT any forbidden terms>,
  "advice": <string WITHOUT any forbidden terms>
}

Your final output MUST be ONLY a valid JSON array with the above objects, nothing else.
"""


# ------------------------------------------------------------
# LLM CLIENT
# ------------------------------------------------------------

# Initialize Groq or OpenAI client
# client = Groq(api_key="")  # Replace with your Groq/OpenAI key


def generate_message_json(top1_problem, top1_prob, top2_problem=None, top2_prob=None):
    """Create the JSON-only prompt for the LLM and decode its response."""

    if top2_problem is None:
        user_payload = {
            "top_1_problem": top1_problem,
            "top_1_probability": top1_prob
        }
    else:
        user_payload = {
            "top_1_problem": top1_problem,
            "top_1_probability": top1_prob,
            "top_2_problem": top2_problem,
            "top_2_probability": top2_prob
        }

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_payload)}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=500
    )

    raw_output = response.choices[0].message.content.strip()
    cleaned = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        decoded = json.loads(cleaned)
    except json.JSONDecodeError:
        decoded = {"error": "Invalid JSON", "raw": raw_output}

    return decoded

# ------------------------------------------------------------
# LOAD ML MODEL + LABEL ENCODER
# ------------------------------------------------------------
@st.cache_resource
def load_audio_model():
    return tf.keras.models.load_model("/home/sfp/Documents/Drive/UPV/CursoIA/Parte2/Notebooks/Project/files/yamnet_model.keras")  # CHANGE NAME

@st.cache_resource
def load_label_encoder():
    return joblib.load("/home/sfp/Documents/Drive/UPV/CursoIA/Parte2/Notebooks/Project/files/label_encoder.plk")

model = load_audio_model()
le = load_label_encoder()
class_names = le.classes_



# ------------------------------------------------------------
# LOAD YAMNET FOR EMBEDDINGS
# ------------------------------------------------------------
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

yamnet = load_yamnet()


def extract_yamnet_embedding(file_path):
    """Returns a 1024-dim embedding from raw audio using YAMNet."""
    wav, sr = librosa.load(file_path, sr=16000)
    waveform = wav.astype(np.float32)

    scores, embeddings, spectrogram = yamnet(waveform)
    embedding = np.mean(embeddings.numpy(), axis=0)  # (1024,)
    return embedding



# ------------------------------------------------------------
# FULL PIPELINE: AUDIO → YAMNET → ML MODEL → LLM JSON
# ------------------------------------------------------------
def run_prediction_and_llm(audio_path, context_number):

    # 1. Extract YAMNet embedding
    embedding = extract_yamnet_embedding(audio_path)
    embedding = embedding.reshape(1, -1)  # (1, 1024)

    # 2. Context
    ctx = np.array([[context_number]], dtype=np.int32)

    # 3. Predict
    probs = model.predict([embedding, ctx])[0]

    # 4. Sort predictions
    top_idx = probs.argsort()[::-1]

    top1_id = top_idx[0]
    top1_label = class_names[top1_id]
    top1_prob = float(probs[top1_id])

    # CASE A: Strong top-1
    if top1_prob >= 0.70:
        return generate_message_json(
            top1_problem=top1_label,
            top1_prob=top1_prob
        )

    # CASE B: Return top-2
    top2_id = top_idx[1]
    top2_label = class_names[top2_id]
    top2_prob = float(probs[top2_id])

    return generate_message_json(
        top1_problem=top1_label,
        top1_prob=top1_prob,
        top2_problem=top2_label,
        top2_prob=top2_prob
    )



# ------------------------------------------------------------
# Rendering explanation
# ------------------------------------------------------------
def render_explanation(json_array):
    """Pretty text rendering for the LLM JSON output."""
    text_output = ""

    for entry in json_array:
        text_output += f"### 🛠️ Problem: **{entry['problem']}**\n"
        text_output += f"📊 **Probability:** {entry['probability']:.2f}\n"
        text_output += f"🔥 **Severity:** {entry['severity']}\n\n"

        text_output += f"**Explanation:** {entry['explanation']}\n\n"

        # Lists
        if isinstance(entry.get("possible_causes"), list):
            text_output += "**Possible causes:**\n"
            for item in entry["possible_causes"]:
                text_output += f"- {item}\n"
            text_output += "\n"

        if isinstance(entry.get("recommended_actions"), list):
            text_output += "**Recommended actions:**\n"
            for item in entry["recommended_actions"]:
                text_output += f"- {item}\n"
            text_output += "\n"

        text_output += f"**What to tell the mechanic:** {entry['what_to_tell_the_mechanic']}\n\n"
        text_output += f"💡 **Advice:** {entry['advice']}\n\n"
        text_output += "---\n\n"

    return text_output


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.title("🚗🔧 AI Mechanic Assistant")
st.markdown("""
### 👋 Hello! I'm your AI mechanic assistant.  
Upload or record your car sound, add the context, and I'll help you understand what might be happening.
""")

st.markdown("---")

st.subheader("🎙 Audio Input")
uploaded_audio = st.file_uploader("Upload audio (WAV/MP3)", type=["wav", "mp3"])
recorded_audio = st.audio_input("Or record sound")

audio_source = uploaded_audio or recorded_audio

st.subheader("🧩 Driving Context")
context = st.selectbox(
    "Choose the correct sound context:",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 — Braking state",
        2: "2 — Moving state",
        3: "3 — Startup state",
        4: "4 — Idle state"
    }[x]
)

st.markdown("---")

if st.button("🔍 Analyze Sound"):
    if audio_source is None:
        st.error("❌ Please upload or record an audio file first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_source.read())
            audio_path = tmp.name

        st.info("Analyzing your audio... 🔧🚗")

        result = run_prediction_and_llm(audio_path, context)

        st.success("✅ Diagnosis Ready!")

        # Convert JSON → formatted text
        if isinstance(result, list):
            explanation_text = render_explanation(result)
            st.markdown(explanation_text)
        else:
            st.error("⚠️ LLM returned invalid structure")
            st.write(result)