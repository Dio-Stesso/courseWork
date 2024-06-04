import openai

# Ініціалізація API ключа
openai.api_key = 'api-key-here'

# Функція для генерації тексту за допомогою GPT-3
def generate_text(prompt, max_tokens=50):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()

# Приклад генерації тексту
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
