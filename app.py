from flask import Flask, request, jsonify, send_from_directory
from main import (
    extract_business_logic,
    generate_language_prompts_multi,
    resolve_languages,
    review_language_prompt,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/models", methods=["GET"])
def models():
    """Expose the model list to the frontend so it isn't hardcoded twice."""
    return jsonify({
        "models": [{"id": k, "label": v["label"]} for k, v in SUPPORTED_MODELS.items()],
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
    model = data.get("model", DEFAULT_MODEL)

    if not business_logic.strip() or not language_prompt.strip():
        return jsonify({"error": "Missing business_logic or language_prompt"}), 400

    review_text = review_language_prompt(business_logic, language_prompt, language, model=model)
    return jsonify({"review": review_text})


if __name__ == "__main__":
    app.run(debug=True, port=5000)