from groq import Groq
import os
import dotenv

client = Groq(api_key=dotenv.get_key(".env", "GROQ_KEY"))
completion = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
      {
        "role": "user",
        "content": "Hi can you show me flex attention local attention implementation?"
      }
    ],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=1,
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
