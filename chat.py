import os
from pathlib import Path

from google import genai


def load_api_key(env_path: str = ".env") -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key.strip().strip('"').strip("'")

    env_file = Path(env_path)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() == "GEMINI_API_KEY":
                value = value.strip().strip('"').strip("'")
                os.environ["GEMINI_API_KEY"] = value
                return value

    raise ValueError("GEMINI_API_KEY not found in environment or .env file.")


def main() -> None:
    try:
        api_key = load_api_key()
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model="gemini-2.5-flash")
    except Exception as exc:
        print(f"Setup error: {exc}")
        return

    print("Gemini chat is ready.")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("Chat ended.")
            break

        try:
            response = chat.send_message(user_message)
            print(f"Gemini: {response.text}\n")
        except Exception as exc:
            print(f"Request error: {exc}\n")


if __name__ == "__main__":
    main()
