from langchain_community.llms import Ollama
import langchain
import prompts as p
from langchain_core.prompts import ChatPromptTemplate
import time

# langchain.debug = True

start = time.time()
llm = Ollama(temperature=0.0, model="llama2")


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", p.system_prompt1),
        ("human", "{text}"),
    ]
)


prompt = prompt_template.format_messages(
    format_instruction=p.format_instruction, text=p.email3
)
# chain = prompt_template | llm


# response = chain.invoke({"text": email2})

response = llm.invoke(prompt)

output_dict = p.output_parser.parse(response)


print(output_dict)

time_used = time.time() - start
print(f"Time used:{time_used:.2f} s")
