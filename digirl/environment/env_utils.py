import torch
from tqdm import tqdm
import numpy as np
import accelerate
from digirl.models import timeout
import requests
import json
import google.generativeai as genai
import re
import PIL.Image
import os

def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


#从总命令得到分步小命令
def decompose_instruction(goal):
    os.environ['http_proxy'] = 'http://127.0.0.1:7900'
    os.environ['https_proxy'] = 'http://127.0.0.1:7900'
    gemini_key = "AIzaSyA_OjTF-CTX3ifa_0U5tJY-mrvEy5n6ZMA"
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # 构造 prompt
    prompt = (
        "你现在是一个手机软件使用专家。我需要你帮我把一条操作手机软件的指令分解成多阶段的分步指令，例如：\n"
        "原指令：打开微信，查看我和第一个通讯录好友的聊天记录。\n"
        "分解指令：(1)打开微信\n"
        "         (2)点击按钮通讯录\n"
        "         (3)点进第一个通讯录好友的对话框\n"
        "         (4)查看和这个好友的聊天记录\n\n"
        "原指令：在小红书上查找关于护肤品的测评，然后对第一条结果点赞。\n"
        "分解指令：(1)打开小红书\n"
        "         (2)点击搜索框\n"
        "         (3)输入“护肤品的测评”\n"
        "         (4)对第一条搜索结果点赞\n\n"
        f"原指令：{goal}\n"
        "分解指令："
    )

    response = model.generate_content(prompt)
    instructions = re.findall(r"\(\d+\)\s(.*?)\n", response.text)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    return instructions


def Is_single_finished(current_goal, image_path):
    os.environ['http_proxy'] = 'http://127.0.0.1:7900'
    os.environ['https_proxy'] = 'http://127.0.0.1:7900'
    gemini_key = "AIzaSyA_OjTF-CTX3ifa_0U5tJY-mrvEy5n6ZMA"
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are an expert in completing tasks based on screenshots and instructions. "
        "I will provide you with a mobile screenshot and a goal. "
        "I want you to determine whether this goal is completed based on the screenshot. "
        f"The goal is: {current_goal}\n"
        "You can only output a single number, either 0 or 1. If you believe the screenshot shows that the current_goal has been completed, output 1; otherwise, output 0."
    )
    screenshot = PIL.Image.open(image_path)  
    response = model.generate_content([prompt, screenshot])
    print(response.text)

    result = int(response.text)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    return result

def batch_interact_environment(agent, env, num_trajectories,\
        accelerator, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, gamma = 0.95, iter=0):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    # broadcast the batch size
    bsize = torch.Tensor([0,]).to(accelerator.device)
    if accelerator.is_main_process:
        bsize[0] = env.bsize
    accelerate.utils.broadcast(bsize)
    bsize = int(bsize.item())
    all_trajectories = []
    if accelerator.is_main_process:
        if hasattr(agent, "critic"):
            env.feature_extractor.model = env.feature_extractor.model.to(env.device)
            agent.critic.to("cpu")
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        if accelerator.is_main_process:
            env.emulator_group_offset = iter * num_trajectories + num_t * bsize
        for _ in range(3):
            try:
                done = False
                trajectories = [[] for _ in range(bsize)]
                #handle the case where the reset fails and timeouts
                reset_success = torch.Tensor([False,]).to(accelerator.device)
                while not all(reset_success):
                    for _ in range(5):
                        try:
                            if accelerator.is_main_process:
                                with timeout(seconds=480): # change this if frequently timeout
                                    batch_obs = env.reset()
                                #the observation space is now a tuple of (text, image)
                                if type(batch_obs[0]['image_feature']) == torch.Tensor:
                                    batch_img = [obs["image_feature"] for obs in batch_obs]
                                else:
                                    batch_img = ["Image feature is not a tensor" for _ in range(bsize)]
                                #if env.feature_extractor is not None:
                                    # colorful_print("autoui has critic, so batch_obs being refractored", "red")
                                #    batch_obs = [obs["prompt"] for obs in batch_obs]
                                reset_success[0] = True
                            accelerate.utils.broadcast(reset_success)
                            break
                        except Exception as e:
                            print(f"Error in environment reset")
                            print(e)
                            if hasattr(env, "reset_appium"):
                                print("Resetting appium")
                                env.reset_appium()
                            accelerate.utils.broadcast(reset_success)
                            continue
                batch_done = torch.Tensor([False,]*bsize).to(accelerator.device)
                accelerate.utils.broadcast(batch_done)
                steps = 0

                for obs in batch_obs:
                    obs['list'] = decompose_instruction(obs['task'])
                    obs['now_step'] = 0

                while not all(batch_done):
                    steps += 1
                    if accelerator.is_main_process:
                        print(f"Environment steps {str(steps)}")
                        print("getting actions!")
                        if env.feature_extractor is not None:
                            results = agent.get_action(batch_obs, torch.cat([i.unsqueeze(0) for i in batch_img], dim = 0))

                        else:
                            results = agent.get_action(batch_obs, None)
                        # import IPython; IPython.embed(); exit(1)
                        action=[]
                        for obs, res in zip(batch_obs, results):
                            obs['score'] = res['score']
                            obs['osatlas_action'] = res['osatlas_action']
                            obs['gemini_action'] = res['gemini_action']
                            if(obs['score'])>=80:
                                action.append(obs['osatlas_action'])
                            else:
                                action.append(obs['gemini_action'])
                        
                        print(batch_obs[0])


                        with timeout(seconds=5*60):
                            batch_return = env.step(decode_f(action))


                        for i,result in zip(range(bsize), batch_return):
                            if result is None:
                                batch_done[i] = True
                                continue
                            obs_dict, r, done = result
                            
                            #先把有用的信息存下来，这样截图路径更新
                            trajectories[i].append({
                                    "goal": batch_obs[i]['task'], \
                                    "current_goal": batch_obs[i]['list'][batch_obs[i]['now_step']], \
                                    "image_path": batch_obs[i]['image_path'], \
                                    "gemini_action": batch_obs[i]['gemini_action'], \
                                    "osatlas_action": batch_obs[i]['osatlas_action'],\
                                    "done": done, \
                                    "score": batch_obs[i]['score']
                                    })
                            batch_obs[i]["image_path"] = obs_dict["image_path"]
                            #截图路径更新完，看当前步数要不要更新
                            print(batch_obs[i]['now_step'])
                            current_finished = Is_single_finished(batch_obs[i]['list'][batch_obs[i]['now_step']], batch_obs[i]['image_path'])

                            if(current_finished==1):
                               batch_obs[i]['now_step'] = batch_obs[i]['now_step']+1
                            
                            #最后存上现在的action能不能完成当前任务
                            trajectories[i].append({
                                    "current_done": current_finished, 
                                    })

                            batch_done[i] = done
                    accelerate.utils.broadcast(batch_done)
                    # print("waiting for everyone")
                    # accelerator.wait_for_everyone()
                    # obs = next_obs
                if accelerator.is_main_process:
                    print(trajectories[0][-1]["next_observation"])
                    all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory), gamma=gamma))\
                                        for trajectory in trajectories]
                break
            except Exception as e:
                print(f"Error in environment interaction")
                import traceback
                print(traceback.format_exc())
                print(e)
                if hasattr(env, "reset_appium"):
                    print("Resetting appium")
                    env.reset_appium()
                continue
    if accelerator.is_main_process:
        if env.feature_extractor is not None:
            env.feature_extractor.model = env.feature_extractor.model.to("cpu")
            if hasattr(agent, "critic"):
                agent.critic.to(agent.device)
        
    return all_trajectories
