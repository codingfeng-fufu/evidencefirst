import config                                                                                                         
from openai import OpenAI                                                                                             
client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.LLM_BASE_URL)                                          
resp = client.chat.completions.create(                                                                                
      model=config.LLM_MODEL,                                                                                           
      messages=[{'role':'user','content':'Reply: FINAL ANSWER: yes'}],                                                  
      max_tokens=20, temperature=0                                                                                      
)                                                                                                                     
print(resp.choices[0].message.content)          