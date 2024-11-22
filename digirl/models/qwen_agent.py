import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from digirl.models.critic import VLMDoubleCritic, TrajectoryCritic
import requests
import json
import google.generativeai as genai
import re
import PIL.Image

def get_gemini_action(final_goal, current_goal, image_path):
    os.environ['http_proxy'] = 'http://127.0.0.1:7900'
    os.environ['https_proxy'] = 'http://127.0.0.1:7900'
    gemini_key = "AIzaSyA_OjTF-CTX3ifa_0U5tJY-mrvEy5n6ZMA"
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
    "You are an expert in completing tasks based on screenshots and instructions. "
    "I will provide you with a mobile screenshot, a final goal, the current goal. "
    "Based on the mobile screenshot, the final goal, and the current goal, I need you to determine the action to take. "
    f"Previous Actions and final goal：{final_goal}\n"
    f"current goal：{current_goal}\n"
    """
    Your skill set includes both basic and custom actions:
    1. Basic Actions
    Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
    Basic Action 1: CLICK 
        - purpose: Click at the specified position.
        - format: CLICK <point>[[x-axis, y-axis]]</point>
        - example usage: CLICK <point>[[101, 872]]</point>
       
    Basic Action 2: TYPE
        - purpose: Enter specified text at the designated location.
        - format: TYPE [input text]
        - example usage: TYPE [Shanghai shopping mall]

    Basic Action 3: SCROLL
        - purpose: SCROLL in the specified direction.
        - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
        - example usage: SCROLL [UP]
    
    2. Custom Actions
    Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

    Custom Action 1: PRESS_BACK
        - purpose: Press a back button to navigate to the previous screen.
        - format: PRESS_BACK
        - example usage: PRESS_BACK

    Custom Action 2: PRESS_HOME
        - purpose: Press a home button to navigate to the home page.
        - format: PRESS_HOME
        - example usage: PRESS_HOME

    Custom Action 3: ENTER
        - purpose: Press the enter button.
        - format: ENTER
        - example usage: ENTER
    
    Custom Action 4: IMPOSSIBLE
        - purpose: Indicate the task is impossible.
        - format: IMPOSSIBLE
        - example usage: IMPOSSIBLE

    Custom Action 5: COMPLETE
        - purpose: Indicate the task is finished.
        - format: COMPLETE
        - example usage: COMPLETE\n\n"""
    "Your output must strictly follow the format below (where gemini_action must be one of the action formats I provided):\n"
    "{action: gemini_action}"
     )

    screenshot = PIL.Image.open(image_path)
    response = model.generate_content([prompt,screenshot])
    #match = re.search(r'gemini_action:\s*(.+)', response.text)
    #action = match.group(1).strip()
    start = response.text.find(':') + 2
    end = response.text.rfind('}')
    action = response.text[start:end]
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    return action
    

def get_gemini_score(final_goal, current_goal, image_path, osatlas_action):
    os.environ['http_proxy'] = 'http://127.0.0.1:7900'
    os.environ['https_proxy'] = 'http://127.0.0.1:7900'   
    gemini_key = "AIzaSyA_OjTF-CTX3ifa_0U5tJY-mrvEy5n6ZMA"
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are an expert in completing tasks based on screenshots and instructions. "
        "I will provide you with a mobile screenshot, a final goal, the current goal and an action. "
        "I hope you evaluate this action based on the screenshot and the goal, giving it a score from 0 to 100. "
        "A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot. "
        f"Previous Actions and final goal：{final_goal}\n"
        f"current goal：{current_goal}\n"
        f"action:{osatlas_action}\n"
        "Your output must strictly follow the format below:\n"
        "{score: }"
    )
    screenshot = PIL.Image.open(image_path)  
    response = model.generate_content([prompt, screenshot])
    start = response.text.find(':') + 2
    end = response.text.rfind('}')
    result = response.text[start:end]
    score = int(result)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    return score

class QwenAgent:
    def __init__(self, device, accelerator, policy_lm=None, critic_lm=None,
            cache_dir='~/.cache', dropout=0.5, TEMPLATE=None, use_lora=False,
                 do_sample=True, temperature=1.0, max_new_tokens=32, use_bfloat16=False, eos_str=None):
        # 加载模型和处理器
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            policy_lm,  torch_dtype="auto", device_map="balanced"
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.trajectory_critic = TrajectoryCritic(device, accelerator, critic_lm=critic_lm, cache_dir=cache_dir, in_dim=768, out_dim=1)
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str

  
    def prepare(self): 
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)
        self.trajectory_critic = self.accelerator.prepare(self.trajectory_critic)

    def _get_a_action(self, obs):
        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
       
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

        Basic Action 3: SCROLL
            - purpose: SCROLL in the specified direction.
            - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - example usage: SCROLL [UP]
    
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: ENTER
            - purpose: Press the enter button.
            - format: ENTER
            - example usage: ENTER

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

        Custom Action 5: COMPLETE
            - purpose: Indicate the task is finished.
            - format: COMPLETE
            - example usage: COMPLETE

        In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.
        Your output must strictly follow the format below , and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas_action must be one of the action formats I provided):
        osatlas_action:
        And your previous actions, current task instruction, and associated screenshot are as follows:
        Previous Actions and final goal: {obs['task']}
        current goal: {obs['list'][obs['now_step']]}
        Screenshot: 
    """
                 
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": sys_prompt,
                            },
                            {
                                "type": "image",
                                "image": obs['image_path'],
                            },
                        ],
                    }
                ]

        # 处理输入并生成
        chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        self.device = self.model.device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128).to(self.device)
        generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

        prefix = 'actions:\n'
        start_index = output_text[0].find(prefix) + len(prefix)
        result = output_text[0][start_index:]
        print(result)
        return result

    def get_action(self, observation, image_features):
        results = []
        #print(len(observation))
        #print(observation)
        for obs in observation:
            
            gemini_action = get_gemini_action(obs['task'], obs['list'][obs['now_step']], obs['image_path'])
            osatlas_action = self._get_a_action(obs)
            score = get_gemini_score(obs['task'], obs['list'][obs['now_step']], obs['image_path'], osatlas_action)
            results.append({
            "gemini_action": gemini_action,
            "osatlas_action": osatlas_action,
            "score": score
             })

        #print(results)
        return results
