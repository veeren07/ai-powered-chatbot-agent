"""
00 - Setup Check

Run this FIRST. It proves your machine is ready before you start the labs:
Python, the packages, your API keys, one real chat call and one graph built
and run end to end.

RUN:
    uv run 00_setup_check.py
"""

import os
import sys

print("====================================================================")
print("MISSION 6 - SETUP CHECK")
print("====================================================================")

problems = []

if sys.version_info < (3, 11):
    problems.append("Python 3.11 or newer is needed. You have " + sys.version.split()[0] + ".")
    print("[!!] Python", sys.version.split()[0], "is too old.")
else:
    print("[OK] Python version is fine.")

try:
    from dotenv import load_dotenv
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    import gradio
    print("[OK] Packages are installed (uv sync worked).")
except Exception as error:
    problems.append("Packages are missing. Run: uv sync   (details: " + str(error) + ")")
    print("[!!] Packages are missing.")
    print("====================================================================")
    print("Run  uv sync  in this folder, then run this file again.")
    raise SystemExit()

load_dotenv()

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    print("[OK] Found the .env file in this folder.")
else:
    problems.append("No .env file in this folder. Create one next to these lab files.")
    print("[!!] No .env file in this folder.")

deepseek = os.getenv("DEEPSEEK_API_KEY", "").strip()
openai = os.getenv("OPENAI_API_KEY", "").strip()

if deepseek != "" and deepseek != "your_deepseek_api_key_here":
    print("[OK] DEEPSEEK_API_KEY found (chat).")
else:
    print("[--] No DEEPSEEK_API_KEY. We will use OpenAI for chat instead.")

if openai != "":
    print("[OK] OPENAI_API_KEY found (chat backup).")
else:
    print("[--] No OPENAI_API_KEY. That is fine as long as DeepSeek works.")

if deepseek == "" and openai == "":
    problems.append("No chat key at all. Add DEEPSEEK_API_KEY (or OPENAI_API_KEY) to .env.")
    print("[!!] No chat key at all.")

# Build and run a tiny graph. This is the real check for this mission.
if len(problems) == 0:
    print("[..] Building a two-node graph and running it...")
    try:
        class CheckState(TypedDict):
            message: str

        def first(state):
            return {"message": state["message"] + " -> first"}

        def second(state):
            return {"message": state["message"] + " -> second"}

        builder = StateGraph(CheckState)
        builder.add_node("first", first)
        builder.add_node("second", second)
        builder.add_edge(START, "first")
        builder.add_edge("first", "second")
        builder.add_edge("second", END)
        graph = builder.compile()
        result = graph.invoke({"message": "start"})
        print("[OK] Graph ran:", result["message"])
    except Exception as error:
        problems.append("Building the graph failed: " + str(error))
        print("[!!] Building the graph failed.")

# Can we draw a graph in the terminal?
if len(problems) == 0:
    try:
        graph.get_graph().draw_ascii()
        print("[OK] Graph drawing works. You will see diagrams in every lab.")
    except Exception:
        print("[--] Graph drawing is unavailable. The labs still work without it.")

if len(problems) == 0:
    import models
    print("[..] Sending one small message to", models.which_provider(), "...")
    try:
        reply = models.get_model(max_tokens=5).invoke("Reply with exactly: ready")
        print("[OK]", models.which_provider(), "replied:", reply.content.strip())
    except Exception as error:
        message = str(error)
        problems.append("The chat call failed: " + message)
        print("[!!] The chat call failed.")
        if "missing_scope" in message:
            print("     Your API key is restricted. Create a new one with All permissions.")
        if "insufficient_quota" in message or "credit" in message:
            print("     That account has no credit left. Top it up or use another key.")

print("====================================================================")
if len(problems) == 0:
    print("ALL GOOD - you are ready.")
    print("Open 01_nodes_and_edges.py next.")
else:
    print("PLEASE FIX THESE:")
    number = 1
    for problem in problems:
        print("  " + str(number) + ". " + problem)
        number = number + 1
print("====================================================================")
