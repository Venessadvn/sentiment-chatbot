from flask import Flask, render_template, request, jsonify, send_file, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
import time

# -------------------------
# HF cache + environment
# -------------------------
os.environ["HF_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# -------------------------
# Flask app + global state
# -------------------------
app = Flask(__name__)
conversation = []  # each item: {"message": str, "sentiment": "Positive"/"Negative", "confidence": float, "emotion": "sad"/...}
MODEL_NAME = "SreyaDvn/sentiment-model"  # <-- set to your correct model id
SAMPLE_UPLOADED_FILE = "/mnt/data/Sentiment_Analysis (2).pdf"  # path you provided earlier

# -------------------------
# Device + model load
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# -------------------------
# Text cleaning & emotion rules
# -------------------------
def clean_text(text):
    t = str(text).strip()
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# simple keyword sets for emotion detection (rule-based)
EMOTION_KEYWORDS = {
    "sad": {"sad", "sadness", "depressed", "depressing", "unhappy", "hurt", "lonely", "feel low", "down", "cry", "tears"},
    "angry": {"angry", "mad", "furious", "pissed", "hate", "annoyed", "irritated", "what the fuck", "wtf"},
    "anxious": {"anxious", "anxiety", "scared", "fear", "worried", "nervous", "panic", "stressed", "stress"},
    "bored": {"bored", "boring", "nothing to do"},
    "pleasant": {"happy", "great", "good", "awesome", "nice", "love", "excited", "glad", "better"}
}

def detect_emotion(text):
    t = clean_text(text)
    # check multi-word triggers first
    for emo, kwset in EMOTION_KEYWORDS.items():
        for kw in kwset:
            if kw in t:
                return emo
    # fallback: simple heuristics
    if "?" in text and len(text.split()) <= 3:
        return "confused"
    return "neutral"

# -------------------------
# Sentiment prediction
# -------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = float(probs[pred])
    return sentiment, confidence

# -------------------------
# Mood shift detection
# -------------------------
def compute_mood_shift(convo):
    if not convo:
        return "No messages yet."
    n = len(convo)
    # use first 30% vs last 30% (min 1)
    win = max(1, math.ceil(n * 0.3))
    first_slice = convo[:win]
    last_slice = convo[-win:]
    def avg_sentiment(slice_):
        # Positive = +1, Negative = -1
        vals = [1 if m["sentiment"] == "Positive" else -1 for m in slice_]
        return sum(vals) / len(vals) if vals else 0
    a_first = avg_sentiment(first_slice)
    a_last = avg_sentiment(last_slice)
    # thresholds small buffer
    delta = a_last - a_first
    if delta > 0.2:
        return "Your mood is improving 😊"
    elif delta < -0.2:
        return "You seem to be getting upset 😔"
    else:
        return "Your mood stayed mostly stable."

# -------------------------
# Trend plot generation
# -------------------------
def make_sentiment_plot(convo):
    # X: message index, Y: +1 for positive, -1 for negative
    if not convo:
        fig, ax = plt.subplots(figsize=(6,2.5))
        ax.text(0.5, 0.5, "No messages yet", ha="center", va="center")
    else:
        ys = [1 if m["sentiment"] == "Positive" else -1 for m in convo]
        xs = list(range(1, len(ys)+1))
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(xs, ys, marker="o")
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 1])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Message #")
        ax.set_title("Sentiment Timeline")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        # fill between to show trend
        ax.fill_between(xs, ys, where=[y>0 for y in ys], alpha=0.08)
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html", sample_uploaded_file=SAMPLE_UPLOADED_FILE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("message", "")
    if not user_text:
        return jsonify({"error":"empty message"}), 400

    sentiment, confidence = predict_sentiment(user_text)
    emotion = detect_emotion(user_text)

    # store
    conversation.append({
        "message": user_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "emotion": emotion
    })

    # emotion + sentiment aware replies
    # base replies by emotion (more specific)
    if emotion == "sad":
        bot_reply = "I'm sorry you're feeling low. Do you want to talk about what's making you feel this way?"
    elif emotion == "angry":
        bot_reply = "That sounds really frustrating. Would you like to tell me more about it?"
    elif emotion == "anxious":
        bot_reply = "I can hear that you're stressed. Try taking a few deep breaths. Do you want to share what's causing it?"
    elif emotion == "bored":
        bot_reply = "Maybe try a quick activity you enjoy. What do you usually like to do?"
    elif emotion == "pleasant":
        bot_reply = "That's great! I'm happy for you. Want to share more?"
    elif emotion == "confused":
        bot_reply = "I might have missed something — can you explain a bit more?"
    else:
        # fallback uses sentiment
        if sentiment == "Negative":
            bot_reply = "I'm sorry you feel that way. I'm here to listen."
        else:
            bot_reply = "That's good to hear. Tell me more."

    # compute mood shift (short summary) to show in UI
    mood_summary = compute_mood_shift(conversation)

    return jsonify({
        "reply": bot_reply,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "emotion": emotion,
        "mood_summary": mood_summary
    })

@app.route("/plot_sentiment.png")
def plot_sentiment_png():
    buf = make_sentiment_plot(conversation)
    return send_file(buf, mimetype="image/png", download_name="sentiment_timeline.png", as_attachment=False)

@app.route("/mood")
def mood():
    return jsonify({"mood_summary": compute_mood_shift(conversation)})

@app.route("/history")
def history():
    return jsonify(conversation)

@app.route("/download_report")
def download_report():
    if not conversation:
        return "Conversation is empty.", 400

    # build plot and save to bytes
    plot_buf = make_sentiment_plot(conversation)

    # build PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title & summary
    story.append(Paragraph("<b>Conversation Sentiment Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))

    positives = sum(1 for m in conversation if m["sentiment"] == "Positive")
    negatives = len(conversation) - positives
    final_sentiment = "Positive" if positives > negatives else "Negative"
    mood_summary = compute_mood_shift(conversation)

    summary_html = f"""
    <b>Total Messages:</b> {len(conversation)}<br/>
    <b>Positive Messages:</b> {positives}<br/>
    <b>Negative Messages:</b> {negatives}<br/>
    <b>Final Sentiment:</b> {final_sentiment}<br/>
    <b>Mood Summary:</b> {mood_summary}
    """
    story.append(Paragraph(summary_html, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Insert plot image into PDF
    # ReportLab wants a file-like object with a .read() method -> use BytesIO
    plot_buf.seek(0)
    # give RLImage the BytesIO directly
    story.append(RLImage(plot_buf, width=450, height=150))
    story.append(Spacer(1, 12))

    # Chat transcript with emotion
    story.append(Paragraph("<b>Chat Transcript</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    for i, m in enumerate(conversation, start=1):
        txt = f"""
        <b>Message {i}:</b> {m['message']}<br/>
        Sentiment: {m['sentiment']}<br/>
        Emotion: {m.get('emotion','') }<br/>
        Confidence: {round(m['confidence'],3)}<br/><hr/>
        """
        story.append(Paragraph(txt, styles["Normal"]))
        story.append(Spacer(1, 6))

    # Optional: show path to uploaded sample file (as provided)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Referenced File (uploaded):</b>", styles["Heading3"]))
    story.append(Paragraph(SAMPLE_UPLOADED_FILE, styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    # return PDF
    ts = int(time.time())
    return send_file(buffer, as_attachment=True, download_name=f"sentiment_report_{ts}.pdf", mimetype="application/pdf")

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
