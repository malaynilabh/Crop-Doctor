"""
Prompt construction for Crop Doctor.

Kept free of any LLM-client or Streamlit code on purpose — these are pure
string-building functions so they're easy to unit-test and reuse.
"""

LANGUAGE_OPTIONS = ["English", "Hindi", "German", "Spanish"]

_LANGUAGE_NAMES = {
    "English": "English",
    "Hindi": "Hindi, written in Devanagari script",
    "German": "German",
    "Spanish": "Spanish",
}


def _language_name(language: str) -> str:
    return _LANGUAGE_NAMES.get(language, "English")


# ------------------------------------------------------------
# 1. DIAGNOSIS EXPLANATION
# ------------------------------------------------------------

def build_system_prompt(language: str = "English") -> str:
    lang = _language_name(language)
    return f"""
You are Crop Doctor, an AI agricultural assistant.

Your role is to explain plant disease predictions produced by an
image classification model.

The image classifier provides the predicted crop disease and its
model probability score.

You must:
1. Explain the predicted disease in simple language.
2. Describe common symptoms of the disease.
3. Explain possible contributing factors to the disease.
4. Provide practical prevention guidance.
5. Clearly distinguish between the model's image-based prediction and confirmed agricultural diagnosis.
6. Avoid presenting uncertain information as fact.
7. Recommend consultation with a qualified agricultural expert when the situation is consequential or uncertain.

Do not invent information about the uploaded image. Use only the diagnosis supplied by the image classifier.

Respond ONLY in {lang} for every text field in your structured output — no
other language, no mixed languages, no translation notes. The field NAMES
in the JSON schema stay exactly as given; only the field VALUES you write
should be in {lang}.
""".strip()


def build_diagnosis_prompt(crop: str, disease: str, confidence: float) -> str:
    return f"""
The image classification model produced the following prediction:

Crop: {crop}
Predicted disease: {disease}
Model probability: {confidence:.2%}

Provide an agricultural explanation covering:

- Diagnosis
- Simple explanation
- Common symptoms
- Possible causes or contributing conditions
- Immediate next steps
- Prevention

The response should be practical and easy for a farmer to understand.
""".strip()


# ------------------------------------------------------------
# 2. SELF-REVIEW AGENT (critic + reviser)
# ------------------------------------------------------------

def build_review_prompt(explanation, crop: str, disease: str, confidence: float, language: str) -> str:
    lang = _language_name(language)
    return f"""
Review this AI-generated plant-disease explanation before it reaches a farmer.

Crop: {crop}
Predicted disease: {disease}
Model probability: {confidence:.2%}
Required language: {lang}

Explanation to review (JSON):
{explanation.model_dump_json(indent=2)}

Check every one of these:
1. Every text field is written entirely in {lang} — no mixed languages.
2. The content is consistent with the predicted disease ({disease}) — it does not invent a different diagnosis.
3. immediate_steps and prevention are practical and specific, not vague filler like "take care of your plant".
4. uncertainty honestly flags that this is a model prediction, not a confirmed lab diagnosis, and suggests expert consultation if warranted.
5. No field is empty or a placeholder.

Return your verdict.
""".strip()


def build_revision_prompt(explanation, feedback: str, crop: str, disease: str, confidence: float, language: str) -> str:
    lang = _language_name(language)
    return f"""
The following plant-disease explanation had issues and needs to be corrected.

Crop: {crop}
Predicted disease: {disease}
Model probability: {confidence:.2%}
Required language: {lang}

Original explanation (JSON):
{explanation.model_dump_json(indent=2)}

Issue(s) to fix: {feedback}

Rewrite the FULL explanation, fixing these issues, in the same structured
format covering diagnosis, disease_meaning, symptoms, possible_causes,
immediate_steps, prevention, and uncertainty.
""".strip()


# ------------------------------------------------------------
# 3. EXPERT CHAT
# ------------------------------------------------------------

def build_chat_system_prompt(crop: str, disease: str, confidence: float, language: str) -> str:
    lang = _language_name(language)
    return f"""
You are 'Crop Doctor', a warm, senior agricultural expert chatting with a
farmer inside a plant-health app.

Respond ONLY in {lang} — no other language, regardless of what language the
farmer writes in.

The farmer's plant was just diagnosed by a CNN model as: {disease} (crop:
{crop}, model probability {confidence:.2%}). Ground every answer in this
diagnosis unless the farmer clearly asks about something else.

Be concise (2-5 sentences, or a short list if the question calls for steps),
practical, and avoid heavy jargon. If asked something unrelated to plant
health or agriculture, gently steer back to the crop issue.
""".strip()