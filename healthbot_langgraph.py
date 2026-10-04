import json
import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph


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
# LangGraph State
# --------------------------------------------------

class HealthBotState(TypedDict):
    health_topic: str
    search_results: str
    summary: str
    quiz_set: list
    answers: list
    feedback: list
    score: int
    continue_session: bool


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

            print(
                f"\nRate limit reached during {task_name}. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    return ""


# --------------------------------------------------
# Build medical summary
# --------------------------------------------------

def build_summary(topic: str, search_results: str) -> str:

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
Create exactly 3 multiple-choice questions using only the
summary below.

Each question must have exactly 4 choices labeled A, B, C, and D.

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

    raw = call_llm(prompt, "quiz generation")

    parsed = parse_quiz_json(raw)

    cleaned = []

    for item in parsed[:3]:

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

        default_options = {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        }

        valid_options = {}

        for key in ["A", "B", "C", "D"]:

            value = str(
                options.get(
                    key,
                    default_options[key]
                )
            ).strip()

            valid_options[key] = (
                value or default_options[key]
            )

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

    return cleaned[:3]


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
Use only the summary below to grade the answer.

Tell the user whether the answer was correct or incorrect.

Mention the correct option and explain it using evidence
from the summary.

Summary:
{summary}

Question:
{question}

Options:
{options}

User Selected Option:
{user_choice}

Correct Option:
{correct_option}
"""

    return call_llm(prompt, "grading")


# --------------------------------------------------
# Node: Get health topic
# --------------------------------------------------

def get_health_topic(
    state: HealthBotState
) -> HealthBotState:

    topic = input(
        "What health topic or condition would you "
        "like to learn about? "
    ).strip()

    state["health_topic"] = topic

    return state


# --------------------------------------------------
# Node: Search medical information
# --------------------------------------------------

def search_medical_info(
    state: HealthBotState
) -> HealthBotState:

    print(
        f"\nSearching trusted sources for: "
        f"{state['health_topic']}..."
    )

    search_results = search_tool.invoke(
        {
            "query": (
                f"{state['health_topic']} "
                "medical information reputable sources"
            )
        }
    )

    state["search_results"] = str(search_results)

    return state


# --------------------------------------------------
# Node: Summarize
# --------------------------------------------------

def summarize_results(
    state: HealthBotState
) -> HealthBotState:

    print("\nGenerating patient-friendly summary...")

    state["summary"] = build_summary(
        state["health_topic"],
        state["search_results"],
    )

    print("\nHEALTH SUMMARY\n")
    print(state["summary"])

    return state


# --------------------------------------------------
# Node: Generate quiz
# --------------------------------------------------

def generate_quiz(
    state: HealthBotState
) -> HealthBotState:

    print("\nGenerating quiz questions...")

    questions = build_quiz_set(
        state["summary"]
    )

    state["quiz_set"] = (
        questions if questions else []
    )

    return state


# --------------------------------------------------
# Node: Present quiz
# --------------------------------------------------

def present_quiz(
    state: HealthBotState
) -> HealthBotState:

    if not state.get("quiz_set"):

        print(
            "Unable to generate a valid quiz set. "
            "Please try another topic."
        )

        state["answers"] = []

        return state

    answers = []

    for i, item in enumerate(
        state["quiz_set"],
        start=1
    ):

        print(
            f"\nQUESTION {i}: "
            f"{item['question']}"
        )

        for key in ["A", "B", "C", "D"]:

            print(
                f"{key}. "
                f"{item['options'][key]}"
            )

        choice = input(
            "\nSelect A, B, C, or D: "
        ).strip().upper()

        while choice not in ["A", "B", "C", "D"]:

            choice = input(
                "Invalid choice. "
                "Please enter A, B, C, or D: "
            ).strip().upper()

        answers.append(
            {
                "question": item["question"],
                "options": item["options"],
                "correct_option": item["correct_option"],
                "user_choice": choice,
            }
        )

    state["answers"] = answers

    return state


# --------------------------------------------------
# Node: Grade quiz
# --------------------------------------------------

def grade_quiz(
    state: HealthBotState
) -> HealthBotState:

    if not state.get("answers"):

        state["feedback"] = []
        state["score"] = 0

        return state

    feedback = []
    score = 0

    for idx, answer in enumerate(
        state["answers"],
        start=1
    ):

        result = grade_mcq(
            state["summary"],
            answer["question"],
            answer["options"],
            answer["user_choice"],
            answer["correct_option"],
        )

        feedback.append(
            {
                "question_number": idx,
                "user_choice": answer["user_choice"],
                "correct_option": answer["correct_option"],
                "feedback": result,
            }
        )

        if (
            answer["user_choice"]
            == answer["correct_option"]
        ):
            score += 1

    state["feedback"] = feedback
    state["score"] = score

    print("\nRESULTS\n")

    for item in feedback:

        print(
            f"Question {item['question_number']}: "
            f"your answer = {item['user_choice']}"
        )

        print(item["feedback"])

        if (
            item["user_choice"]
            == item["correct_option"]
        ):

            print("Correct!")

        else:

            print(
                "Incorrect. Correct answer: "
                f"{item['correct_option']}"
            )

    print(
        f"\nFinal score: "
        f"{score}/{len(state['quiz_set'])}"
    )

    return state


# --------------------------------------------------
# Node: Ask whether to continue
# --------------------------------------------------

def ask_continue(
    state: HealthBotState
) -> HealthBotState:

    choice = input(
        "\nWould you like to learn about another "
        "health topic? (yes/no): "
    ).strip().lower()

    state["continue_session"] = (
        choice in ["yes", "y"]
    )

    return state


# --------------------------------------------------
# Node: Reset
# --------------------------------------------------

def reset_state(
    state: HealthBotState
) -> HealthBotState:

    return {
        "health_topic": "",
        "search_results": "",
        "summary": "",
        "quiz_set": [],
        "answers": [],
        "feedback": [],
        "score": 0,
        "continue_session": False,
    }


# --------------------------------------------------
# Build LangGraph workflow
# --------------------------------------------------

workflow = StateGraph(HealthBotState)

workflow.add_node(
    "get_topic",
    get_health_topic
)

workflow.add_node(
    "search",
    search_medical_info
)

workflow.add_node(
    "summarize",
    summarize_results
)

workflow.add_node(
    "quiz",
    generate_quiz
)

workflow.add_node(
    "present_quiz",
    present_quiz
)

workflow.add_node(
    "grade",
    grade_quiz
)

workflow.add_node(
    "ask_continue",
    ask_continue
)

workflow.add_node(
    "reset",
    reset_state
)


# --------------------------------------------------
# Workflow edges
# --------------------------------------------------

workflow.add_edge(
    "get_topic",
    "search"
)

workflow.add_edge(
    "search",
    "summarize"
)

workflow.add_edge(
    "summarize",
    "quiz"
)

workflow.add_edge(
    "quiz",
    "present_quiz"
)

workflow.add_edge(
    "present_quiz",
    "grade"
)

workflow.add_edge(
    "grade",
    "ask_continue"
)


# --------------------------------------------------
# Conditional routing
# --------------------------------------------------

def should_continue(
    state: HealthBotState
) -> str:

    if state.get("continue_session", False):
        return "reset"

    return "__end__"


workflow.add_conditional_edges(
    "ask_continue",
    should_continue,
    {
        "reset": "reset",
        "__end__": "__end__",
    },
)

workflow.add_edge(
    "reset",
    "get_topic"
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

workflow.set_entry_point("get_topic")

graph = workflow.compile()


# --------------------------------------------------
# Initial state
# --------------------------------------------------

initial_state = {
    "health_topic": "",
    "search_results": "",
    "summary": "",
    "quiz_set": [],
    "answers": [],
    "feedback": [],
    "score": 0,
    "continue_session": False,
}


# --------------------------------------------------
# Start HealthBot
# --------------------------------------------------

print(
    "HealthBot Initialized. "
    "Starting workflow...\n"
)

graph.invoke(initial_state)