import json
import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(".env")

mistral_api_key = os.getenv("MISTRAL_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not mistral_api_key:
    raise ValueError(
        "MISTRAL_API_KEY is missing. Add it to .env."
    )

if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Add it to .env."
    )


# --------------------------------------------------
# Rate limiter
# --------------------------------------------------

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


# --------------------------------------------------
# Mistral LLM
# --------------------------------------------------

llm = ChatMistralAI(
    model="ministral-14b-2512",
    api_key=mistral_api_key,
    temperature=0.7,
    max_retries=2,
    timeout=120,
    rate_limiter=rate_limiter,
)


# --------------------------------------------------
# Tavily Search
# --------------------------------------------------

search_tool = TavilySearchResults(
    api_key=tavily_api_key
)


# --------------------------------------------------
# LLM helper
# --------------------------------------------------

def call_llm(prompt: str, task_name: str) -> str:

    for attempt in range(1, 4):

        try:
            response = llm.invoke(prompt)
            return response.content

        except Exception as exc:

            message = str(exc).lower()

            if "429" not in message and "rate limit" not in message:
                raise

            if attempt == 3:
                raise

            wait_time = 2 ** attempt

            st.warning(
                f"Rate limit reached during {task_name}. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    return ""


# --------------------------------------------------
# Build summary
# --------------------------------------------------

def build_summary(
    topic: str,
    search_results: str
) -> str:

    prompt = f"""
Use only the medical information below to write a 3-4 paragraph
summary in plain, patient-friendly language.

Do not use outside knowledge.
Do not invent facts.
Focus on clarity, safety, and common patient understanding.

Topic:
{topic}

Medical Information:
{search_results}
"""

    return call_llm(prompt, "summarize")


# --------------------------------------------------
# Parse quiz JSON
# --------------------------------------------------

def parse_quiz_json(raw_text):

    cleaned = (raw_text or "").strip()

    if not cleaned:
        return []

    if cleaned.startswith("```"):

        cleaned = (
            cleaned
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    try:

        parsed = json.loads(cleaned)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1 and end > start:

        candidate = cleaned[start:end + 1]

        try:

            parsed = json.loads(candidate)

            if isinstance(parsed, list):
                return parsed

        except Exception:
            pass

    return []


# --------------------------------------------------
# Generate quiz
# --------------------------------------------------

def build_quiz_set(summary: str):

    prompt = f"""
Create exactly ONE multiple-choice quiz question using only
the summary below.

The question must have exactly 4 choices labeled A, B, C, and D.

Mark the correct answer with the field "correct_option"
using one of A, B, C, or D.

Return valid JSON only in this exact structure:

[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_option": "A"
  }}
]

Do not include anything outside the JSON array.

Summary:
{summary}
"""

    raw = call_llm(
        prompt,
        "quiz generation"
    )

    parsed = parse_quiz_json(raw)

    cleaned = []

    for item in parsed[:1]:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        if not question:
            continue

        options = item.get("options", {})

        if not isinstance(options, dict):
            options = {}

        valid_options = {}

        for key in ["A", "B", "C", "D"]:
            valid_options[key] = str(
                options.get(key, f"Option {key}")
            ).strip()

        correct_option = str(
            item.get("correct_option", "A")
        ).upper()

        if correct_option not in ["A", "B", "C", "D"]:
            correct_option = "A"

        cleaned.append(
            {
                "question": question,
                "options": valid_options,
                "correct_option": correct_option,
            }
        )

    return cleaned[:1]


# --------------------------------------------------
# Grade MCQ
# --------------------------------------------------

def grade_mcq(
    summary: str,
    question: str,
    options: dict,
    user_choice: str,
    correct_option: str,
) -> str:

    prompt = f"""
You are grading a patient education comprehension check.

Use ONLY the summary below to evaluate the patient's answer.

SUMMARY:
{summary}

QUESTION:
{question}

OPTIONS:
{options}

PATIENT'S ANSWER:
{user_choice}

CORRECT ANSWER:
{correct_option}

Assign one grade:

A = Completely correct
B = Mostly correct
C = Incorrect

Provide a brief justification based ONLY on the summary.

Return exactly this format:

Grade: A/B/C
Justification: <brief explanation based only on the summary>
"""

    return call_llm(
        prompt,
        "grading"
    )


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="HealthBot",
    page_icon="💊",
    layout="centered",
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title("💊 HealthBot")

st.caption(
    "AI-powered health education assistant"
)


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz_set" not in st.session_state:
    st.session_state.quiz_set = []

if "responses" not in st.session_state:
    st.session_state.responses = {}

if "feedback" not in st.session_state:
    st.session_state.feedback = {}


# --------------------------------------------------
# Health topic form
# --------------------------------------------------

with st.form("health_form"):

    topic = st.text_input(
        "What health topic or condition "
        "would you like to learn about?",
        value=st.session_state.topic,
    )

    submitted = st.form_submit_button(
        "Search and summarize"
    )


# --------------------------------------------------
# Search and generate summary
# --------------------------------------------------

if submitted and topic.strip():

    st.session_state.topic = topic.strip()

    with st.spinner(
        "Searching trusted sources and summarizing..."
    ):

        search_results = search_tool.invoke(
            {
                "query": (
                    f"{st.session_state.topic} "
                    "medical information reputable sources"
                )
            }
        )

        summary = build_summary(
            st.session_state.topic,
            str(search_results),
        )

        quiz_set = build_quiz_set(
            summary
        )

    st.session_state.summary = summary

    st.session_state.quiz_set = quiz_set

    st.session_state.responses = {}

    st.session_state.feedback = {}


# --------------------------------------------------
# Display summary
# --------------------------------------------------

if st.session_state.summary:

    st.subheader("Health Summary")

    st.write(
        st.session_state.summary
    )


# --------------------------------------------------
# Comprehension check readiness
# --------------------------------------------------

#if st.session_state.summary:
if st.session_state.summary and st.session_state.get("ready_check") == "Yes":

    st.subheader("Comprehension Check")

    ready = st.radio(
        "Are you ready for the comprehension check?",
        ["Yes", "No"],
        index=None,
        key="ready_check",
    )

    if ready == "No":
        st.info("Please review the summary and select Yes when you are ready.")

# --------------------------------------------------
# Display quiz
# --------------------------------------------------

if st.session_state.summary:

    st.subheader("Quiz")

    if not st.session_state.quiz_set:

        st.warning(
            "Unable to generate a valid quiz set. "
            "Please try another topic."
        )

    else:

        for idx, item in enumerate(
            st.session_state.quiz_set
        ):

            question = item.get(
                "question",
                ""
            )

            options = item.get(
                "options",
                {
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D",
                },
            )

            correct = item.get(
                "correct_option",
                "A"
            )

            st.markdown(
                f"### Question {idx + 1}"
            )

            st.write(question)

            option_labels = [
                f"{key}. {options.get(key, f'Option {key}')}"
                for key in ["A", "B", "C", "D"]
            ]

            choice_label = st.radio(
                "Select one answer",
                option_labels,
                key=f"q_{idx}",
                index=None,
                horizontal=False,
            )

            choice = (
                choice_label[0]
                if choice_label
                else None
            )

            st.session_state.responses[idx] = choice

            if choice:

                if st.button(
                    f"Check Answer {idx + 1}",
                    key=f"check_{idx}",
                ):

                    with st.spinner(
                        "Checking your answer..."
                    ):

                        st.session_state.feedback[idx] = (
                            grade_mcq(
                                st.session_state.summary,
                                question,
                                options,
                                choice,
                                correct,
                            )
                        )

            if idx in st.session_state.feedback:

                st.write(
                    st.session_state.feedback[idx]
                )

                if choice == correct:

                    st.success("Correct!")

                else:

                    st.error(
                        f"Incorrect. "
                        f"The correct answer is {correct}."
                    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("HealthBot")

    if st.button("Reset"):

        st.session_state.topic = ""

        st.session_state.summary = ""

        st.session_state.quiz_set = []

        st.session_state.responses = {}

        st.session_state.feedback = {}

        st.rerun()

    st.info(
        "This app searches the web and summarizes "
        "medical information in plain language."
    )