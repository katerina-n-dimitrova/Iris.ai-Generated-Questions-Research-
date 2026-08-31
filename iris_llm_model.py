from openai import OpenAI



if __name__ == '__main__':

    client = OpenAI(
        base_url="https://chat-api-dev.iris.ai/v1/",
        api_key="EMPTY",  # if auth isn't required, something like "dummy" often works
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-4B",  # must match the name/alias exposed by vLLM
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain prefix tuning in two sentences."
            }
        ],
        temperature=0.2,
        max_tokens=200,
    )

    print(response.choices[0].message.content)