from flask import Flask, request, jsonify, send_from_directory
from main import (
    extract_business_logic,
    generate_language_prompts_multi,
    resolve_languages,
    review_language_prompt,
    apply_review_fixes,
    is_flow_json,
    extract_text_from_flow_json,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    DEFAULT_REVIEW_MODEL,
)

app = Flask(__name__, static_folder="static")


@app.errorhandler(Exception)
def handle_any_error(e):
    """Catch-all so the frontend always gets JSON back, never Flask's default
    HTML error page — that HTML is what causes 'Unexpected token <' on the
    frontend when something breaks server-side (bad API key, SDK error, etc.)."""
    import traceback
    traceback.print_exc()  # still shows the real traceback in Render logs
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/models", methods=["GET"])
def models():
    """Expose the model list to the frontend so it isn't hardcoded twice."""
    return jsonify({
        "models": [{"id": k, "label": v["label"], "group": v.get("group", "Flash")} for k, v in SUPPORTED_MODELS.items()],
        "default": DEFAULT_MODEL,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json or {}
    raw_prompt = data.get("raw_prompt", "")
    mode = data.get("mode")  # "extract" or "scoped"
    requested_languages = data.get("languages", [])
    model = data.get("model", DEFAULT_MODEL)

    if model not in SUPPORTED_MODELS:
        return jsonify({"error": f"Unknown model '{model}'"}), 400

    if not raw_prompt.strip():
        return jsonify({"error": "Prompt is empty"}), 400

    # If the pasted/uploaded input is a conversation-flow JSON export, flatten it
    # into a plain-text prompt first — everything downstream (extraction, scoping,
    # language generation) then runs exactly as it would on a hand-typed prompt.
    flow_json = is_flow_json(raw_prompt)
    if flow_json:
        raw_prompt = extract_text_from_flow_json(flow_json)
        if not raw_prompt.strip():
            return jsonify({"error": "Flow JSON was recognized but contained no usable prompt text"}), 400

    languages = resolve_languages(requested_languages)
    if not languages:
        return jsonify({"error": "No valid languages provided"}), 400

    if mode == "extract":
        business_logic = extract_business_logic(raw_prompt, model=model)
    else:
        business_logic = raw_prompt  # already clean — passed through untouched, never re-processed

    result = generate_language_prompts_multi(business_logic, languages, model=model)

    return jsonify({
        "business_logic": business_logic,
        "language_prompts": result["prompts"],
        "warnings": result["warnings"],  # {lang: [violation strings]} — only present for langs with issues
        "model": model,
    })


@app.route("/api/review", methods=["POST"])
def review():
    data = request.json or {}
    business_logic = data.get("business_logic", "")
    language_prompt = data.get("language_prompt", "")
    language = data.get("language", "")

    if not business_logic.strip() or not language_prompt.strip():
        return jsonify({"error": "Missing business_logic or language_prompt"}), 400

    # Review always uses the stronger model — catching subtle contradictions matters
    # more here than matching whatever model the user picked for generation.
    review_text = review_language_prompt(business_logic, language_prompt, language, model=DEFAULT_REVIEW_MODEL)
    return jsonify({"review": review_text, "model": DEFAULT_REVIEW_MODEL})


@app.route("/api/apply_review", methods=["POST"])
def apply_review():
    data = request.json or {}
    business_logic = data.get("business_logic", "")
    language_prompt = data.get("language_prompt", "")
    review_text = data.get("review_text", "")
    language = data.get("language", "")

    if not business_logic.strip() or not language_prompt.strip() or not review_text.strip():
        return jsonify({"error": "Missing business_logic, language_prompt, or review_text"}), 400

    fixed_prompt = apply_review_fixes(business_logic, language_prompt, review_text, language, model=DEFAULT_REVIEW_MODEL)
    return jsonify({"language_prompt": fixed_prompt, "model": DEFAULT_REVIEW_MODEL})


if __name__ == "__main__":
    app.run(debug=True, port=5000)