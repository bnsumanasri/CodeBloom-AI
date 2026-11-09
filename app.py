
from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "llama3.2:3b"
MAX_CODE_LENGTH = 15000
REQUEST_TIMEOUT = 240

SUPPORTED_LANGUAGES = {
    "Python",
    "Java",
    "JavaScript",
    "C",
    "C++"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()

        tags_data = response.json()
        available_models = [
            item.get("name", "")
            for item in tags_data.get("models", [])
        ]

        model_available = any(
            name == MODEL or name.startswith(MODEL + ":")
            for name in available_models
        )

        return jsonify({
            "status": "ok" if model_available else "warning",
            "app": "AI Code Explainer",
            "ollama": "connected",
            "model": MODEL,
            "modelAvailable": model_available,
            "languages": sorted(SUPPORTED_LANGUAGES)
        })

    except requests.Timeout:
        return jsonify({
            "status": "error",
            "ollama": "timeout",
            "error": "Ollama did not respond in time."
        }), 503

    except requests.ConnectionError:
        return jsonify({
            "status": "error",
            "ollama": "not connected",
            "error": "Start Ollama and try again."
        }), 503

    except (requests.RequestException, ValueError, TypeError):
        return jsonify({
            "status": "error",
            "ollama": "unavailable",
            "error": "Could not check Ollama status."
        }), 503


@app.route("/api/explain", methods=["POST"])
def explain_code():
    if not request.is_json:
        return jsonify({
            "error": "Please send the code as JSON."
        }), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request. Please try again."
        }), 400

    code = data.get("code", "")
    language = data.get("language", "Python")

    if not isinstance(code, str) or not code.strip():
        return jsonify({
            "error": "Please enter some code."
        }), 400

    if len(code) > MAX_CODE_LENGTH:
        return jsonify({
            "error": "Code is too long. Please keep it under 15,000 characters."
        }), 400

    if not isinstance(language, str):
        return jsonify({
            "error": "Please select a valid language."
        }), 400

    language = language.strip()

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "error": "Unsupported language. Select Python, Java, JavaScript, C, or C++."
        }), 400

    prompt = f"""
Act as a precise, beginner-friendly {language} programming tutor.

Analyze this code and use these five headings:

1. SIMPLE EXPLANATION
Explain the program's purpose in 2-4 sentences.

2. LINE-BY-LINE EXPLANATION
Explain each important statement briefly, in order.

3. EXPECTED OUTPUT
Trace variable values, conditions, and loops carefully.
Calculate arithmetic and distinguish strings from numbers.
Show the expected output in a code block.
If the output depends on input or environment, state the assumption.
Never claim that you executed the program.

4. ERRORS AND BUGS
Identify only genuine errors or plausible risks supported by the code.
If none are apparent, say so.

5. IMPROVED CODE
Give valid {language} code only if a useful improvement is needed.
Preserve the program's purpose and briefly explain changes.
Otherwise, say that no improvement is necessary.

Be concise but sufficiently detailed for a beginner.
Do not invent output, bugs, or missing information.
Ensure your explanation, calculations, and code agree.

Code:
```{language}
{code}
```
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.2,
                    "num_predict": 600
                }
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        result_data = response.json()
        message = result_data.get("message", {})
        result = (
            message.get("content", "")
            if isinstance(message, dict)
            else ""
        )

        if not isinstance(result, str) or not result.strip():
            return jsonify({
                "error": "The AI returned an empty or unexpected response."
            }), 502

        return jsonify({
            "explanation": result,
            "language": language,
            "model": MODEL
        })

    except requests.Timeout:
        return jsonify({
            "error": "The AI took too long. Please try again."
        }), 504

    except requests.ConnectionError:
        return jsonify({
            "error": "Could not connect to Ollama. Check that Ollama is running."
        }), 502

    except requests.HTTPError:
        return jsonify({
            "error": "Ollama returned an error. Check that the model is available."
        }), 502

    except (ValueError, KeyError, TypeError):
        return jsonify({
            "error": "The AI returned an unexpected response."
        }), 502

    except requests.RequestException:
        return jsonify({
            "error": "A network error occurred while contacting Ollama."
        }), 502


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
