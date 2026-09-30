import os
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()


def extract_text(content) -> str:
    """
    Safely extracts a plain string from LLM response .content.
    Gemini can return a list of content parts like [{'type': 'text', 'text': '...'}]
    instead of a plain string. This normalizes it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

def pick_llm(level: str = "medium"):
    """
    Picks the appropriate LLM based on the level of the question.
    Currently configured to use Google Gemini.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, temperature=0)

    # -------------------------------------------------------------------------
    # OpenAI and Anthropic configurations 

    # -------------------------------------------------------------------------
    # if level.lower() == "low":
    #     return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # elif level.lower() in ("medium", "high"):
    #     return ChatOpenAI(model_name="gpt-4o", temperature=0)
    # elif level.lower() == "claude":
    #     return ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0)

    raise ValueError("No API key configured. Please set GEMINI_API_KEY in your .env file.")

if __name__ == "__main__":
    llm_obj = pick_llm("low")  
    print(llm_obj.invoke("What is the capital of France?"))
