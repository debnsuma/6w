import requests
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from bs4 import BeautifulSoup

class Website:
    def __init__(self):
        self.system_message = "You are a helpful assistant"
        self.client = Groq()
        self.model = "llama-3.3-70b-versatile"

    def get_contents(self, url):
        response = requests.get(url)
        body = response.content
        soup = BeautifulSoup(body, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
        return f"Webpage Title:\n{title}\nWebpage Contents:\n{text}\n\n"

    def message_llm(self, prompt):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def get_summary(self, company_name, url):
        prompt = f"Please summarize this website of {company_name}. Here is their landing page:\n"
        prompt += self.get_contents(url)
        result = self.message_llm(prompt)
        return result

    def stream_llm(self, prompt):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result

    def stream_summary(self, company_name, url):
        prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
        prompt += self.get_contents(url)
        result = self.stream_llm(prompt)
        yield from result
    
    def run(self):
        view = gr.Interface(
            fn=self.get_summary,
            inputs=[
                gr.Textbox(label="Company name:"),
                gr.Textbox(label="Landing page URL including http:// or https://"),
            ],
            outputs=[gr.Markdown(label="Brochure:")],
            flagging_mode="never",
        )
        view.launch()

if __name__ == "__main__":
    load_dotenv()
    web = Website()
    web.run()
   

