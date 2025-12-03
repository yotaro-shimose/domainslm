import os
from dotenv import load_dotenv
from agents import set_tracing_export_api_key


def setup_openai_tracing():
    load_dotenv()
    set_tracing_export_api_key(os.environ["OPENAI_API_KEY"])
