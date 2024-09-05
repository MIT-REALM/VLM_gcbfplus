import datetime
import functools as ft
import os
import jax
import jax.numpy as jnp
import numpy as np
import json
import pytictoc
import re 
import time 
import json, boto3
import matplotlib.pyplot as plt
import base64
import math
import scipy

os.environ['AWS_PROFILE'] = "MyProfile1"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"



from gcbfplus.env.utils import get_lidar

from ..utils.graph import GraphsTuple

from ..utils.utils import tree_stack

from .rrt_utils import find_path

from .A_star_utils import AStarPlanner

from .kmeans_utils import kmeans

astar = AStarPlanner(1, 2)

from openai import OpenAI

t = pytictoc.TicToc()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_init_prompt(num_incontext, preset=False, model='llm'):
    """
    Create the initial prompt for the LLM model.
    Args:
        num_incontext: the number of in-context examples
        preset: whether to use a preset prompt
    Returns:
        log_message: the log message
        log_dir: the log directory
    """
    
    

    file_name = 'LLM_traj_tests'
    if preset:
        log_dir = f"{'LLM_files'}/num_incontext_{num_incontext}/{file_name}"
    else:
        log_dir = f"{'LLM_files'}/random_num_incontext_{num_incontext}/{file_name}"
    os.makedirs(log_dir, exist_ok=True)
    if model == 'llm':
        prompt_file = 'LLM_files/LLM_in_context_prompts_llm.txt'
    else:
        prompt_file = 'LLM_files/LLM_in_context_prompts_vlm.txt'

    with open(prompt_file, "r") as f:
        prompt = f.read()

    prompts = prompt.split("\n")

    log_message = []
    system_message = {"role": "system", "content": "You are a helpful assistant for multi-robot deadlock resolution situations designed to provide an answer as a JSON object."}
    log_message.append(system_message)
    for p in prompts:
        log_message.append({"role": "user", "content": p})
        
    return log_message, log_dir

class LLM_utils:
    def __init__(        self,
        num_agents,
        num_waypoints=0,
        use_local_leader=False,
        use_normalized_data=False,
        leader_model=None,
        LLM_calls=1,
        leader_assign_count=0,
        keep_mode=20,
        t_mode=0,
        num_runtime_incontext_prompts=0,
        num_incontext_prompts=0,
        log_dir=None,
        use_cot=False,
        collect_incontext_data=False,
        collect_vlm_incontext_data=False,
        vlm_example_prompts= None,
        use_N_obs=-1,
        area_size=None,
        use_rrt=False,
    ):  
        self.num_agents = num_agents
        self.n_agent = num_agents
        self.num_waypoints = num_waypoints
        self.use_local_leader = use_local_leader
        self.use_normalized_data = use_normalized_data
        self.leader_model = leader_model
        self.LLM_calls = LLM_calls
        self.leader_assign_count = leader_assign_count
        self.t_mode = t_mode
        self.keep_mode = keep_mode
        self.log_dir = log_dir
        self.num_runtime_incontext_prompts = num_runtime_incontext_prompts
        self.num_incontext_prompts = num_incontext_prompts
        self.use_cot = use_cot
        self.collect_incontext_data = collect_incontext_data
        self.collect_vlm_incontext_data = collect_vlm_incontext_data
        self.vlm_example_prompts = vlm_example_prompts
        self.xm = []
        self.xM = []
        self.ym = []
        self.yM = []
        self.all_obs = []
        self.all_graphs = []
        self.far_agent_indices = jnp.arange(num_agents)
        self.waypoints = -1
        self.fig_count = 0
        self.leaders = -1
        self.assignments = jnp.zeros((num_agents,)).astype(jnp.int32)
        self.main_leader = -1
        self.iter = 0
        self.use_N_obs = use_N_obs
        self.area_size = area_size
        self.use_RRT = use_rrt
        
        if self.collect_incontext_data:
            incontext_file = 'LLM_files/RRT_LLM_in_context_prompts_' + str(num_agents) + '.txt'
            
            self.incontext_file = incontext_file
        else:
            self.incontext_file = None

    def find_samples(self, graph, num_agents, leader, all_obs=[]):
        
        agent_states = graph.type_states(0, num_agents)
        goals = graph.xg
        leader_state = agent_states[leader.item()]
        leader_goal = goals[leader.item()]
        if len(all_obs) == 0:
            obs_center = jnp.array([self.xM + self.xm, self.ym + self.yM]) / 2
            obs_center = obs_center.T
            obs_width = jnp.array([self.xM - self.xm]) * 0.9 / 2
            obs_height = jnp.array([self.yM - self.ym]) * 0.9 / 2
        else:
            obs_center = np.array(all_obs)
            obs_width = np.ones(len(all_obs)) * 0.1
            obs_width = obs_width[None, :] / 2
            obs_height = np.ones(len(all_obs)) * 0.1
            obs_height = obs_height[None, :] / 2
        
        _, ax = plt.subplots()
        
        ax.set_xlim(0, 5.1)
        ax.set_ylim(0, 5.1)
        ax.set_aspect('equal')
        ax.scatter(agent_states[:, 0], agent_states[:, 1], c='b', label='Agent')
        ax.scatter(goals[:, 0], goals[:, 1], c='g', label='Goal')
        
        
        for i in range(len(obs_center)):
            ax.add_patch(plt.Rectangle((obs_center[i][0] - obs_width[0, i] / 2, obs_center[i][1] - obs_height[0, i] / 2), obs_width[0, i], obs_height[0, i], fill=True, color='k'))
        
        plt.legend()
        plt.tight_layout()
        plt.grid('both')
        fig_path = self.log_dir
        
        plt.savefig(f"{fig_path}/sample_{self.fig_count}.png")
        plt.close()
        
        t.tic()
        path = find_path(obs_center, obs_width, obs_height, leader_state, leader_goal, max_samples=10000000)
        
        if path is None:
            c= np.arange(0, 1, 0.1)
            points = np.array([leader_state + c[i] * (leader_goal - leader_state) for i in range(len(c))])
            self.fig_count += 1
            sample_time = t.tocvalue()
            return points, 0, sample_time
        points = np.array(path)
        sample_time = t.tocvalue()
        
        self.fig_count += 1
            
        return path, 1, sample_time

    def find_samples_astar(self, graph, num_agents, leader, all_obs=[], all_obs_x=[], all_obs_y=[]):
        
        agent_states = graph.type_states(0, num_agents)
        goals = graph.xg
        leader_state = agent_states[leader.item()]
        leader_goal = goals[leader.item()]
        if len(all_obs) == 0:
            obs_center = jnp.array([self.xM + self.xm, self.ym + self.yM]) / 2
            obs_center = obs_center.T
            obs_width = jnp.array([self.xM - self.xm]) * 0.9
            obs_height = jnp.array([self.yM - self.ym]) * 0.9
        else:
            obs_center = np.array(all_obs)
            obs_width = np.ones(len(all_obs)) * 0.1
            obs_width = obs_width[None, :]
            obs_height = np.ones(len(all_obs)) * 0.1
            obs_height = obs_height[None, :]
        
        # min_x_obs_coordinate = np.min(obs_center[:, 0] - obs_width[0, :])
        # min_y_obs_coordinate = np.min(obs_center[:, 1] - obs_height[0, :])
        
        # max_x_obs_coordinate = np.max(obs_center[:, 0] + obs_width[0, :]) + 1 - min_x_obs_coordinate
        # max_y_obs_coordinate = np.max(obs_center[:, 1] + obs_height[0, :]) + 1 - min_y_obs_coordinate
        _, ax = plt.subplots()
        
        # ax.set_xlim(0, 5.1)
        # ax.set_ylim(0, 5.1)
        ax.set_aspect('equal')
        ax.scatter(agent_states[:, 0], agent_states[:, 1], c='b', label='Agent')
        ax.scatter(goals[:, 0], goals[:, 1], c='g', label='Goal')
        # ax.scatter(leader_state[0], leader_state[1], c='b', label='Leader')
        # ax.scatter(leader_goal[0], leader_goal[1], c='g', label='Leader Goal')
        for i in range(len(obs_center)):
            ax.add_patch(plt.Rectangle((obs_center[i][0] - obs_width[0, i] / 2, obs_center[i][1] - obs_height[0, i] / 2), obs_width[0, i], obs_height[0, i], fill=True, color='k'))
        
        # This is a grid world environment with obstacles shown in black rectangular blocks. Can you provide a concise description of the obstacles in a JSON format {"Id": Id, "Location": (x,y), "Length": L, "Orientation": o} where "Location" is the location of the starting of the obstacle, "Length" its length and "Orientation" is horizontal or vertical.  Please provide a description for all the obstacles in the image. 
        
        plt.legend()
        plt.tight_layout()
        plt.grid('both')
        fig_path = self.log_dir
        
        # plt.savefig(f"{fig_path}/sample_{self.fig_count}.png")
        # plt.close()
        # print(Asas)
        t.tic()
        # all_obs_x = all_obs[:, 0]
        # all_obs_y = all_obs[:, 1]
        # breakpoint()
        all_obs_x += [0., self.area_size * 20]
        all_obs_y += [0., self.area_size * 20]
        
        path_x, path_y = astar.planning(int(leader_state[0]*20), int(leader_state[1] * 20), int(leader_goal[0]*20), int(leader_goal[1]*20), all_obs_x, all_obs_y)
        
        # path = find_path(obs_center, obs_width, obs_height, leader_state, leader_goal, max_samples=10000000)
        # path = None
        if path_x is None:
            c= np.arange(0, 1, 0.1)
            points = np.array([leader_state + c[i] * (leader_goal - leader_state) for i in range(len(c))])
            
            # ax.plot(points[:, 0], points[:, 1], c='r', label='Leader Path')
            # plt.legend()
            
            # plt.savefig(f"{fig_path}/sample_{self.fig_count}.png")
            self.fig_count += 1
            sample_time = t.tocvalue()
            return points, 0, sample_time
        else:
            path_x.reverse()
            path_y.reverse()
            path = np.array([path_x, path_y]).T / 20
            # path = path.reverse()
        # points = np.array(path)[:, :self.waypoints]
        points = path
        if len(points) < 5:
            c= np.arange(0, 1, 0.1)
            points_add = np.array([leader_state + c[i] * (leader_goal - leader_state) for i in range(len(c))])
            points = np.concatenate([points, points_add], axis=0)
            
        sample_time = t.tocvalue()
        ax.plot(points[:, 0], points[:, 1], c='r', label='Leader Path')
        
        plt.legend()
        
        plt.savefig(f"{fig_path}/sample_{self.fig_count}.png")
        plt.close()
        self.fig_count += 1
            
        return path, 1, sample_time
    
    def check_response(self, completion):
        completion = completion.replace(' ', '')
        completion = completion.replace('\n', '')
        completion = completion.replace("'", '"')
        if '{' in completion and '}' in completion:
            
            count = len(completion)
            start = completion.index('{')
            while completion[start+1] != '"':
                completion = completion[start+1:]
                start = completion.index('{')
                if len(completion) == 1:
                    count = 0
                    break
                else:
                    count = len(completion)
            if count == 0:
                return completion, count
            end = completion.index('}')
            
            json_strings = completion[start:end+1]
            json_strings = json_strings.replace("\n", '')
            json_strings = json_strings.replace("(", '[')
            json_strings = json_strings.replace(")", ']')
            try:
                _ = json.loads(json_strings)
                count = 10
                print('a json response')
            except:
                print('not a json response')
                count = 0
        else:
            json_strings = ''
            count = 0
        
        return json_strings, count

    def generate_current_grid(self, graph, agent_states=None, goals=None, all_obs=[], leader=-1):
        num_agents = self.num_agents
        if agent_states is None:
            agent_states = graph.type_states(0, num_agents)
        if goals is None:
            goals = graph.xg
        agent_goal_dist = jnp.linalg.norm(agent_states[:, :2] - goals[:, :2], axis=1)
        
        close_agent_mask = jnp.array([True])
        xg = goals
        x = agent_states[:, 0]
        y = agent_states[:, 1]
        if leader == -1:
            if self.use_local_leader and close_agent_mask.any():
                
                
                far_agent_mask = agent_goal_dist > 0.5
                far_agent_indices = jax.numpy.where(far_agent_mask)
                
                no_obs_in_path_agents = []
                
                
                    
                obs_i = jnp.unique(jnp.array(all_obs).round(2), axis=0)
                if len(obs_i) == 0:
                    no_obs_in_path_agents = jax.numpy.arange(num_agents)
                else:
                    for i in range(self.num_agents):
                    
                        obs_dir = obs_i - jnp.array([x[i], y[i]])
                        goal_dir = xg[i] - jnp.array([x[i], y[i]])
                        obs_dir = obs_dir / (jax.numpy.linalg.norm(obs_dir, axis=0) + 1e-6)
                        goal_dir = goal_dir / (jax.numpy.linalg.norm(goal_dir, axis=0) + 1e-6)
                        obs_dot_goal = jax.numpy.dot(obs_dir, goal_dir)
                        obs_dot_goal_max = jax.numpy.max(obs_dot_goal)
                        
                        if obs_dot_goal_max > 0.1:
                            no_obs_in_path_agents.append(i)
                
                agents_for_prompt = jax.numpy.array(no_obs_in_path_agents)

                far_agent_indices = jax.numpy.intersect1d(far_agent_indices[0], agents_for_prompt)
            else:
                far_agent_indices = jax.numpy.arange(num_agents)
        else:
            far_agent_indices = jnp.array([jax.numpy.array([leader]).squeeze()])
            
        far_agent_count = len(far_agent_indices)
        if far_agent_count == 0:
            far_agent_indices = jax.numpy.arange(num_agents)
            
        _, ax = plt.subplots()
        if len(all_obs) == 0:
            obs_center = jnp.array([self.xM + self.xm, self.ym + self.yM]) / 2
            obs_center = obs_center.T
            obs_width = jnp.array([self.xM - self.xm]) * 0.9
            obs_height = jnp.array([self.yM - self.ym]) * 0.9
        else:
            if self.use_N_obs > -1:
                
                all_obs = all_obs[-self.use_N_obs:]
            all_obs = jnp.array(all_obs).round(2)
            all_obs = jnp.unique(all_obs, axis=0).tolist()
            obs_center = np.array(all_obs)
            obs_width = np.ones(len(all_obs)) * 0.1
            obs_width = obs_width[None, :]
            obs_height = np.ones(len(all_obs)) * 0.1
            obs_height = obs_height[None, :]
        agent_states = agent_states[far_agent_indices]
        goals = goals[far_agent_indices]
        
        min_x = np.min(agent_states[:, 0])
        min_x = np.min([min_x, np.min(goals[:, 0])])
        min_x = np.min([min_x, np.min(obs_center[:, 0] - obs_width[0, :])]) - 0.2
        max_x = np.max(agent_states[:, 0])
        max_x = np.max([max_x, np.max(goals[:, 0])])
        max_x = np.max([max_x, np.max(obs_center[:, 0] + obs_width[0, :])]) + 0.2
        min_y = np.min(agent_states[:, 1])
        min_y = np.min([min_y, np.min(goals[:, 1])])
        min_y = np.min([min_y, np.min(obs_center[:, 1] - obs_height[0, :])]) - 0.2
        max_y = np.max(agent_states[:, 1])
        max_y = np.max([max_y, np.max(goals[:, 1])])
        max_y = np.max([max_y, np.max(obs_center[:, 1] + obs_height[0, :])]) + 0.2
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(agent_states[:, 0], agent_states[:, 1], c='b', label='Agent')
        ax.scatter(goals[:, 0], goals[:, 1], c='g', label='Goal')
        for i in range(len(obs_center)):
            ax.add_patch(plt.Rectangle((obs_center[i][0] - obs_width[0, i] / 2, obs_center[i][1] - obs_height[0, i] / 2), obs_width[0, i], obs_height[0, i], fill=True, color='k'))
        
        plt.grid('both')
        fig_path = self.log_dir
        plt.tight_layout()
        fig_path = os.path.join(fig_path, 'current_grid' + str(self.fig_count) + '.jpg')
        
        self.fig_count += 1
        plt.savefig(fig_path,bbox_inches='tight',pad_inches=0, transparent=True)
        
        plt.close()
        return fig_path
    
    def gpt_response(self, prompt, model, LLM_calls=1, all_obs=[], json_output=True, graph=None):
        prompts = []
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        for prompt_i in prompt:
            if type(prompt_i) == str:
                prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"').replace('`', "'"))
            prompts.append(prompt_i)
        client = OpenAI(
            
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        log_dir = self.log_dir
        LLM_time = 0
        if model == 'gpt-4o' or model == 'gpt-4-turbo':
            prompt = prompt[:-1]
            str_prompt = ''
            for p in prompt:
                str_prompt += p['content']
            str_prompt += 'The agent locations and their respective goal locations are described below to help you choose the leader and waypoints.'
            goals = graph.xg
            agent_states = graph.type_states(0, self.n_agent)
            
            if self.main_leader > -1 or self.leaders == -1:
                agent_states = graph.type_states(0, self.n_agent)
                
                if self.main_leader == -1:
                    
                    
                    agent_indices = jnp.arange(self.n_agent)
                    leader = -1
                else:
                    leader = self.main_leader
                    agent_indices = jnp.array([leader])
                
                image = self.generate_current_grid(graph, all_obs=all_obs, leader=leader)
                response, in_tokens, out_tokens, LLM_time = self.gpt_client_vlm(str_prompt, model, client, agent_states, goals, agent_indices, image)
                leader_id, waypoints = self.LLM_leader_waypoint_fn(response, graph, self.num_agents)
                if self.main_leader == -1 and leader_id is not None:
                    leader = leader_id
                    self.main_leader = leader
                if waypoints is None:
                    c = np.arange(0, 1, 0.2)
                    waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
                if len(waypoints) < self.num_waypoints:
                    leader_goal = goals[leader]
                    waypoints = jnp.concatenate([waypoints, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)], axis=0)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(response))
                    f.write('\n')
                waypoints = waypoints[None, ...]
            else:
                in_tokens = 0
                out_tokens = 0
                num_clusters = max(self.assignments) + 1
                waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
                for i in range(num_clusters):
                    agent_indices = jnp.where(self.assignments == i)[0]
                    agent_states_i = agent_states[agent_indices]
                    num_agents = len(agent_indices)
                    goals_i = goals[agent_indices]
                    leader = agent_indices[self.leaders[i]]
                    image = self.generate_current_grid(graph, all_obs=all_obs, leader=leader)
                    response, in_token, out_token, LLM_time_cluster = self.gpt_client_vlm(str_prompt, model, client, agent_states_i, goals_i, agent_indices, image)
                    LLM_time += LLM_time_cluster
                    in_tokens += in_token
                    out_tokens += out_token
                    _, waypoint = self.LLM_leader_waypoint_fn(response, graph, num_agents)
                    if len(waypoint) < self.num_waypoints:
                        leader_goal = goals[agent_indices[self.leaders[i]]]
                        waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                    else:
                        waypoint = waypoint[:self.num_waypoints]
                    waypoints = waypoints.at[i].set(waypoint)
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write(str(response))
                        f.write('\n')
        else:
            if self.main_leader > -1 or self.leaders == -1:
                agent_states = graph.type_states(0, self.n_agent)
                
                if self.main_leader == -1:
                    
                    
                    agent_indices = jnp.arange(self.n_agent)
                    leader = -1
                else:
                    leader = self.main_leader
                    agent_indices = jnp.array([leader])
                goal_states = graph.xg
                goals = goal_states
                response, LLM_time = self.gpt_client_llm(prompt, model, client, json_output, LLM_calls, agent_indices, agent_states, goal_states)
                in_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
                return_message = response.choices[0].message.content
                leader_id, waypoints = self.LLM_leader_waypoint_fn(return_message, graph, self.num_agents)
                if self.main_leader == -1 and leader_id is not None:
                    leader = leader_id
                    self.main_leader = leader
                if waypoints is None:
                    c = np.arange(0, 1, 0.2)
                    waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
                
                if len(waypoints) < self.num_waypoints:
                    leader_goal = goals[leader]
                    waypoints = jnp.concatenate([waypoints, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)], axis=0)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(response))
                    f.write('\n')
                waypoints = waypoints[None, ...]
            else:
                in_tokens = 0
                out_tokens = 0
                num_clusters = max(self.assignments) + 1
                waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
                for i in range(num_clusters):
                    agent_indices = jnp.where(self.assignments == i)[0]
                    agent_states_i = graph.type_states(0, len(agent_indices))
                    goals_i = graph.xg[agent_indices]
                    leader = agent_indices[self.leaders[i]]
                    response, LLM_time_cluster = self.gpt_client_llm(prompt, model, client, json_output, LLM_calls, agent_indices, agent_states_i, goal_states_i)
                    in_tokens += response.usage.prompt_tokens
                    out_tokens += response.usage.completion_tokens
                    return_message = response.choices[0].message.content
                    LLM_time += LLM_time_cluster
                    _, waypoint = self.LLM_leader_waypoint_fn(return_message, graph, len(agent_indices))
                    if len(waypoint) < self.num_waypoints:
                        leader_goal = goals_i[self.leaders[i]]
                        waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                    else:
                        waypoint = waypoint[:self.num_waypoints]
                    waypoints = waypoints.at[i].set(waypoint)
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write(str(response))
                        f.write('\n')
        return waypoints, in_tokens, out_tokens, LLM_time
    
    def gpt_client_llm(self, prompt, model, client, json_output, LLM_calls, agent_indices, agent_states, goal_states):
        prompt = prompt[:-1]
        
        new_prompt = self.get_new_prompt(agent_states, goal_states, agent_indices)

        message = {"role": "user", "content": new_prompt}
        
        prompt.append(message)
        
        if LLM_calls == 1:
            tmp = 0.0
            if 'gpt-4' in model:
                model = 'gpt-4-turbo'
        else:
            if 'gpt-4' in model:
                model = 'gpt-4-turbo'
                tmp = 0.7
            else:
                tmp = 1.0
        print('prompting LLM with model: ', model, ' and temperature: ', tmp)
        t.tic()
        if tmp == 0.0:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=tmp,
                seed=100,
                response_format={"type": "json_object"} if json_output else None,
                max_tokens=1000,
                
            )
        else:    
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=tmp,
                
                response_format={"type": "json_object"} if json_output else None,
                max_tokens=1000,
                
            )
        LLM_time = t.tocvalue()
        return response, LLM_time
    
    def gpt_client_vlm(self, str_prompt, model, client, agent_states, goals, agent_indices, image):
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        j = 0
        for i in agent_indices:
            x = agent_states[i, 0]
            y = agent_states[i, 1]
            str_prompt += '***AgentId***' + str(j + 1) + '***current state***(' + '{:.1f}'.format(x.item()) + ',' + '{:.1f}'.format(y.item()) + ')'
            str_prompt += '***goal location***(' + '{:.1f}'.format(goals[i, 0].item()) + ',' + '{:.1f}'.format(goals[i, 1].item()) + ')'
        
        base64_image = encode_image(image)
        attempt = 0
        t.tic()
        completion = ''
        while attempt < 10:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str_prompt},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                            },
                        ],
                        }
                    ],
                    temperature=0.0,
                    seed=100,
                    response_format={"type": "json_object"},
                    max_tokens=1000,
                    )
                attempt = 100
                response_["prompt_token_count"] += response.usage.prompt_tokens
                response_["generation_token_count"] += response.usage.completion_tokens
                completion = response.choices[0].message.content
                break
            except:
                attempt += 1
            
        
        LLM_time = t.tocvalue()
        return completion, response_["prompt_token_count"], response_["generation_token_count"], LLM_time
        
    def get_new_prompt(self, agent_states, goals, agent_indices):
        new_prompt = ''
        xm = self.xm
        xM = self.xM
        ym = self.ym
        yM = self.yM
        if self.use_N_obs > -1:
            xm = xm[-self.use_N_obs:]
            xM = xM[-self.use_N_obs:]
            ym = ym[-self.use_N_obs:]
            yM = yM[-self.use_N_obs:]
            
        obs = jnp.array([xm, ym])
        obs = obs.T
        obs_w = xM - xm
        obs_h = yM - ym
            
        iter = self.iter
        x = agent_states[:, 0]
        y = agent_states[:, 1]
        xg = goals
        new_prompt = '***Env:' + str(iter)
        new_prompt += '***Number of agents***1' 
        new_prompt +='***Safety radius***0.05***Connectivity radius***0.5***'
        new_prompt += 'Agents'
        j = 0
        far_agent_indices = agent_indices.tolist()
        for i in far_agent_indices:
            new_prompt += '***AgentId***' + str(j + 1) + '***current state***(' + '{:.2f}'.format(x[i].item()) + ',' + '{:.2f}'.format(y[i].item()) + ')***goal location***(' + '{:.2f}'.format(xg[i][0].item()) + ',' + '{:.2f}'.format(xg[i][0].item()) + ')' 
            j += 1
            
        new_prompt += '***obstacles***['
        obs_for_prompts = obs
        
        
        if type(obs_for_prompts) == list:
            obs_for_prompts = jnp.array(obs_for_prompts)
        len_obs = len(obs_for_prompts)
        for i in range(len_obs):
            new_prompt += '(x=' + '{:.2f}'.format(obs_for_prompts[i][0].item()) + ', y=' + '{:.2f}'.format(obs_for_prompts[i][1].item())
            new_prompt +=  ', w=' + '{:.2f}'.format(obs_w[i].item()) + ',h=' + '{:.2f}'.format(obs_h[i].item()) + ')'
            if i < len_obs - 1:
                new_prompt += ','

        if new_prompt[-1] == ',':
            new_prompt = new_prompt[:-1]
            
        if new_prompt[-1] != '[':
            new_prompt += ']'
        else:
            new_prompt += 'None]'
        
        new_prompt += '***Number of waypoints***' + str(self.num_waypoints)

        return new_prompt
    
    def llama_response(self, graph, prompt, model):
        
        prompt = prompt[:-1]
        log_dir = self.log_dir
        str_prompts = ''
        for prompt_i in prompt:
            if type(prompt_i) == str:
                prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                prompt_i = prompt_i['content'].replace('`', '"')
                str_prompts += prompt_i
            else:
                str_prompts += prompt_i['content']

        agent_states = graph.type_states(0, self.n_agent)
        goals = graph.xg
        LLM_time = 0
        if self.main_leader > -1 or self.leaders == -1:
            if self.main_leader == -1:
                
                
                agent_indices = jnp.arange(self.n_agent)
                leader = -1
            else:
                leader = self.main_leader
                agent_indices = jnp.array([leader])
            
            
            new_prompt = self.get_new_prompt(agent_states, goals, agent_indices)
            
            str_prompts += new_prompt
            str_prompts += '. The answer must be of the form {"Leader": 1, "Waypoints": [[x0, y0], [x1, y1], ...]}. '
            str_prompts += "Please provide the answer as a JSON object with 'Leader' and 'Waypoints' as the keys and a JSON object output, no other explanation or text. What is the leader and waypoint assignment for this environment?"
            str_prompts = str_prompts.replace('*', ' ')
            
            assert type(str_prompts) == str
            
            completion, in_tokens, out_tokens, LLM_time = self.llama_client_llm(str_prompts, model)
            leader_id, waypoints = self.LLM_leader_waypoint_fn(completion, graph, self.num_agents)
            if self.main_leader == -1 and leader_id is not None:
                leader = leader_id
                self.main_leader = leader
            
            if waypoints is None:
                c = np.arange(0, 1, 0.2)
                waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
            if len(waypoints) < self.num_waypoints:
                leader_goal = goals[leader]
                add_waypoints = jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)
                waypoints = jnp.concatenate([waypoints, add_waypoints], axis=0)
                
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write(str(completion))
                f.write('\n')
        else:
            in_tokens = 0
            out_tokens = 0
            num_clusters = max(self.assignments) + 1
            waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
            for i in range(num_clusters):
                prompt_i = str_prompts
                agent_indices = jnp.where(self.assignments == i)[0]
                agent_states_i = agent_states[agent_indices]
                goals_i = goals[agent_indices]
                leader = agent_indices[self.leaders[i]]
                new_prompt = self.get_new_prompt(agent_states_i, goals_i, agent_indices)
                prompt_i += new_prompt
                str_prompts += '. The answer must be of the form {"Leader": 1, "Waypoints": [[x0, y0], [x1, y1], ...]}. '
                str_prompts += "Please provide the answer as a JSON object with 'Leader' and 'Waypoints' as the keys and a JSON object output, no other explanation or text. What is the leader and waypoint assignment for this environment?"
                str_prompts = str_prompts.replace('*', ' ')
                
                assert type(str_prompts) == str
                
                completion, in_token, out_token, LLM_time_clusters = self.llama_client_llm(str_prompts, model)
                in_tokens += in_token
                out_tokens += out_token
                LLM_time += LLM_time_clusters
                _, waypoint = self.LLM_leader_waypoint_fn(completion, graph, len(agent_indices))
                if len(waypoint) < self.num_waypoints:
                    leader_goal = goals_i[self.leaders[i]]
                    waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                else:
                    waypoint = waypoint[:self.num_waypoints]
                waypoints = waypoints.at[i].set(waypoint)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(completion))
                    f.write('\n')
        return waypoints, in_tokens, out_tokens, LLM_time
    
    def llama_client_llm(self, str_prompts, model):
        str_prompts = f"<s>[INST] {str_prompts} [/INST]"

        region ="us-east-1"
        client = boto3.client('bedrock-runtime',region)
        
        count = -1
        num_calls = 0
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        t.tic()
        while count < 5 and num_calls < 10:
            
            print('try number: ', num_calls + 1)
            
            print('Prompting LLM with model: ', model, ' and temperature: ', num_calls / 10)
            body = {"prompt": str_prompts,"temperature": num_calls / 10,"top_p": 0.9,"max_gen_len": 100}
            response = client.invoke_model(modelId=model,body=json.dumps(body), accept="application/json")
            
            num_calls += 1
                        
                        
            response_body = json.loads(response["body"].read())
            count = response_body["generation_token_count"]
            completion = response_body["generation"]
            print('completion: ', completion)
            if len(completion) == 0:
                print('empty completion')
                count = 0
                
            
            completion, count = self.check_response(completion)
            response_['prompt_token_count'] += response_body['prompt_token_count']
            response_['generation_token_count'] += response_body['generation_token_count']
                    
            
            
        return_message = completion
        response = response_body
        response_["generation"] = completion
        LLM_time = t.tocvalue()
        return completion, response_['prompt_token_count'], response_['generation_token_count'], LLM_time
    
    def mistral_response(self, graph, prompt, model):
        prompt = prompt[:-1]
        log_dir = self.log_dir
        str_prompts = ''
        for prompt_i in prompt:
            if type(prompt_i) == str:
                prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                prompt_i = prompt_i['content'].replace('`', '"')
                str_prompts += prompt_i
            else:
                str_prompts += prompt_i['content']
        agent_states = graph.type_states(0, self.n_agent)
        goals = graph.xg
        LLM_time = 0
        if self.main_leader > -1 or self.leaders == -1:
            if self.main_leader == -1:
                
                
                agent_indices = jnp.arange(self.n_agent)
                leader = -1
            else:
                leader = self.main_leader
                agent_indices = jnp.array([leader])
            
            new_prompt = self.get_new_prompt(agent_states, goals, agent_indices)
            str_prompts += new_prompt
            response, in_tokens, out_tokens, LLM_time = self.mistral_client_llm(str_prompts, model)
            leader_id, waypoints = self.LLM_leader_waypoint_fn(response, graph, self.num_agents)
            if self.main_leader == -1 and leader_id is not None:
                leader = leader_id
                self.main_leader = leader
            
            if waypoints is None:
                c = np.arange(0, 1, 0.2)
                waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
            if len(waypoints) < self.num_waypoints:
                leader_goal = goals[leader]
                waypoints = jnp.concatenate([waypoints, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)], axis=0)
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write(str(response))
                f.write('\n')
            waypoints = waypoints[None, ...]
        else:
            in_tokens = 0
            out_tokens = 0
            num_clusters = max(self.assignments) + 1
            waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
            for i in range(num_clusters):
                prompt_i = str_prompts
                agent_indices = jnp.where(self.assignments == i)[0]
                agent_states_i = agent_states[agent_indices]
                goals_i = goals[agent_indices]
                leader = agent_indices[self.leaders[i]]
                new_prompt = self.get_new_prompt(agent_states_i, goals_i, agent_indices)
                prompt_i += new_prompt
                response, in_token, out_token, LLM_time_clusters = self.mistral_client_llm(str_prompts, model)
                in_tokens += in_token
                out_tokens += out_token
                LLM_time += LLM_time_clusters
                _, waypoint = self.LLM_leader_waypoint_fn(response, graph, len(agent_indices))
                if len(waypoint) < self.num_waypoints:
                    leader_goal = goals_i[self.leaders[i]]
                    waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                else:
                    waypoint = waypoint[:self.num_waypoints]
                waypoints = waypoints.at[i].set(waypoint)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(response))
                    f.write('\n')
        
        return waypoints, in_tokens, out_tokens, LLM_time
    
    def mistral_client_llm(self, str_prompts, model):
        str_prompts += '. The answer must be of the form {"Leader": 1, "Waypoints": [[x0, y0], [x1, y1], ...]}. '
        str_prompts += "Please provide the answer as a JSON object with 'Leader' and 'Waypoints' as the keys and a JSON object output, no other explanation or text. What is the leader and waypoint assignment for this environment?"
        str_prompts = str_prompts.replace('*', ' ')
        
        assert type(str_prompts) == str
        
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        
        instruction = f"<s>[INST] {str_prompts} [/INST]"
        body = {
            "prompt": instruction,
            "max_tokens": 200,
            "temperature": 0.5,
        }
        region ="us-east-1"
        client = boto3.client('bedrock-runtime',region)

        count = -1
        num_calls = 0
        t.tic()
        while count < 5 and num_calls < 10:
            print('try number: ', num_calls + 1)
            num_calls += 1
            
            print('Prompting LLM with model: ', model, ' and temperature: ', num_calls / 10)
            response = client.invoke_model(modelId=model, body=json.dumps(body))
            time.sleep(5)
            response_body = json.loads(response["body"].read())
            outputs = response_body.get("outputs")

            completions = [output["text"] for output in outputs]
            completion = completions[0]
            count = len(completion)
            if count == 0:
                print('empty completion')
                continue
            completion, count = self.check_response(completion)
            response_['prompt_token_count'] += int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
            response_['generation_token_count'] += int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
        LLM_time = t.tocvalue()
        response = response_
        return completion, response_['prompt_token_count'], response_['generation_token_count'], LLM_time
    
    def claude2_response(self, graph, prompt, model):
        log_dir = self.log_dir
        if self.num_incontext_prompts + self.num_runtime_incontext_prompts > 0:
            new_prompt = prompt[-1]
            prompt = prompt[:-1]
            example_prompts = prompt[-2*(self.num_incontext_prompts + self.num_runtime_incontext_prompts):]
            prompt = prompt[:-2*(self.num_incontext_prompts + self.num_runtime_incontext_prompts)]
        else:
            example_prompts = []
            new_prompt = prompt[-1]
            prompt = prompt[:-1]
        region ="us-east-1"
        client = boto3.client('bedrock-runtime',region)
        str_prompts = ''
        for prompt_i in prompt:
            if type(prompt_i) == str:
                prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                prompt_i = prompt_i['content'].replace('`', '"')
                str_prompts += prompt_i
            else:
                str_prompts += prompt_i['content']
        str_prompts += '. The answer must be of the form {"Leader": 1, "Waypoints": [[x0, y0], [x1, y1], ...]}. '
        str_prompts += "Please provide the answer as a JSON object with 'Leader' and 'Waypoints' as the keys and a JSON object output, no other explanation or text. What is the leader and waypoint assignment for this environment?"
        str_prompts = str_prompts.replace('*', ' ')
        enclosed_prompt = "Human: " + str_prompts
        LLM_time = 0
        if self.main_leader > -1 or self.leaders == -1:
            agent_states = graph.type_states(0, self.n_agent)
            goals = graph.xg
            if self.main_leader == -1:
                agent_indices = jnp.arange(self.n_agent)
                leader = -1
            else:
                leader = self.main_leader
                agent_indices = jnp.array([leader])

            new_prompt = self.get_new_prompt(agent_states, goals, agent_indices)
            completion, in_tokens, out_tokens, LLM_time = self.claude2_client_llm(model, example_prompts, new_prompt, client, enclosed_prompt)
            leader_id, waypoints = self.LLM_leader_waypoint_fn(completion, graph, self.num_agents)
            if self.main_leader == -1 and leader_id is not None:
                leader = leader_id
                self.main_leader = leader
            if waypoints is None:
                c = np.arange(0, 1, 0.2)
                waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
            if len(waypoints) < self.num_waypoints:
                leader_goal = goals[leader]
                waypoints = jnp.concatenate([waypoints, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)], axis=0)
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write(str(completion))
                f.write('\n')
            waypoints = waypoints[None, ...]
        else:
            in_tokens = 0
            out_tokens = 0
            num_clusters = max(self.assignments) + 1
            waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
            for i in range(num_clusters):
                agent_indices = jnp.where(self.assignments == i)[0]
                agent_states_i = graph.type_states(0, len(agent_indices))
                goals_i = graph.xg[agent_indices]
                leader = agent_indices[self.leaders[i]]
                new_prompt = self.get_new_prompt(agent_states_i, goals_i, agent_indices)
                completion, in_token, out_token, LLM_time_clusters = self.claude2_client_llm(model, example_prompts, new_prompt, client, enclosed_prompt)
                in_tokens += in_token
                out_tokens += out_token
                LLM_time += LLM_time_clusters
                _, waypoint = self.LLM_leader_waypoint_fn(completion, graph, len(agent_indices))
                if len(waypoint) < self.num_waypoints:
                    leader_goal = goals_i[self.leaders[i]]
                    waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                else:
                    waypoint = waypoint[:self.num_waypoints]
                waypoints = waypoints.at[i].set(waypoint)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(completion))
                    f.write('\n')
        return waypoints, in_tokens, out_tokens, LLM_time
    
    def claude2_client_llm(self, model, example_prompts, new_prompt, client, enclosed_prompt):
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        if len(example_prompts) > 0:
            prompt_ind = 0
            for prompt_i in example_prompts:
                if prompt_ind % 2 == 0 and prompt_ind > 0:
                    enclosed_prompt += 'Human: ' 
                elif prompt_ind % 2 == 1:
                    enclosed_prompt += 'Assistant: '
                if type(prompt_i) == str:
                    prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                    prompt_i = prompt_i['content'].replace('`', '"')
                    enclosed_prompt += prompt_i
                else:
                    enclosed_prompt += prompt_i['content']
                prompt_ind += 1
            enclosed_prompt += 'Human: ' + new_prompt
        enclosed_prompt += '\n\nAssistant:'
        
        count = -1
        num_calls = 0
        t.tic()
        while count < 5 and num_calls < 10:
            body = {
                "prompt": enclosed_prompt,
                "max_tokens_to_sample": 1000,
                "temperature": num_calls / 10,
                "stop_sequences": ["\n\nHuman:"],
            }

            response = client.invoke_model(
                modelId=model, body=json.dumps(body)
            )
            time.sleep(5)
            num_calls += 1
            response_body = json.loads(response["body"].read())
            completion = response_body["completion"]
            count = len(completion)
            if len(completion) == 0:
                print('empty completion')
                count = 0
            response_['generation'] = completion
            response_['prompt_token_count'] += int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
            response_['generation_token_count'] += int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
            completion, count = self.check_response(completion)
        LLM_time = t.tocvalue()
        return completion, response_['prompt_token_count'], response_['generation_token_count'], LLM_time
    
    def claude3_response(self, graph, prompt, model):
        log_dir = self.log_dir
        if self.num_incontext_prompts + self.num_runtime_incontext_prompts > 0:
            prompt = prompt[:-1]
            example_prompts = prompt[-2*(self.num_incontext_prompts + self.num_runtime_incontext_prompts):]
            prompt = prompt[:-2*(self.num_incontext_prompts + self.num_runtime_incontext_prompts)]
        else:
            example_prompts = []
            prompt = prompt[:-1]
        region ="us-west-2"
        client = boto3.client('bedrock-runtime',region)
        str_prompts = ''
        LLM_time = 0
        for prompt_i in prompt:
            if type(prompt_i) == str:
                prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                prompt_i = prompt_i['content'].replace('`', '"')
                str_prompts += prompt_i
            else:
                str_prompts += prompt_i['content']

        if self.main_leader > -1 or self.leaders == -1:
            agent_states = graph.type_states(0, self.n_agent)
            goals = graph.xg
            if self.main_leader == -1:
                
                
                agent_indices = jnp.arange(self.n_agent)
                leader = -1
            else:
                leader = self.main_leader
                agent_indices = jnp.array([leader])
            
            new_prompt = self.get_new_prompt(agent_states, goals, agent_indices)
            str_prompts += new_prompt
            response, in_tokens, out_tokens, LLM_time = self.claude3_client_llm(str_prompts, model, client, example_prompts, new_prompt)
            leader_id, waypoints = self.LLM_leader_waypoint_fn(response, graph, self.num_agents)
            if self.main_leader == -1 and leader_id is not None:
                leader = leader_id
                self.main_leader = leader_id
            
            if waypoints is None:
                c = np.arange(0, 1, 0.2)
                waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
            if len(waypoints) < self.num_waypoints:
                leader_goal = goals[leader]
                waypoints = jnp.concatenate([waypoints, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)], axis=0)
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write(str(response))
                f.write('\n')
            waypoints = waypoints[None, ...]
        else:
            in_tokens = 0
            out_tokens = 0
            num_clusters = max(self.assignments) + 1
            waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
            for i in range(num_clusters):
                agent_indices = jnp.where(self.assignments == i)[0]
                agent_states_i = graph.type_states(0, len(agent_indices))
                goals_i = graph.xg[agent_indices]
                leader = agent_indices[self.leaders[i]]
                new_prompt = self.get_new_prompt(agent_states_i, goals_i, agent_indices)
                str_prompts += new_prompt
                response, in_token, out_token, LLM_time_clusters = self.claude3_client_llm(str_prompts, model, client, example_prompts, new_prompt)
                in_tokens += in_token
                out_tokens += out_token
                LLM_time += LLM_time_clusters
                _, waypoint = self.LLM_leader_waypoint_fn(response, graph, len(agent_indices))
                if len(waypoint) < self.num_waypoints:
                    leader_goal = goals_i[self.leaders[i]]
                    waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                else:
                    waypoint = waypoint[:self.num_waypoints]
                waypoints = waypoints.at[i].set(waypoint)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(response))
                    f.write('\n')        
        
        return waypoints, in_tokens, out_tokens, LLM_time
    
    def claude3_client_llm(self, str_prompts, model, client, example_prompts, new_prompt):
        claude_prompt = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": str_prompts}],
                        }
                    ]
        if len(example_prompts) > 0:
            claude_prompt.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": 'Okay, please provide the examples and their correct answers as reference for the new scenario.'}],
                }
            )
            prompt_ind = 0

            for prompt_i in example_prompts:
                str_assistant_prompts = ''
                str_user_prompts = ''
                if type(prompt_i) == str:
                    prompt_i = json.loads(prompt_i.replace('"', "`").replace("'", '"'))
                    prompt_i = prompt_i['content'].replace('`', '"')
                    if prompt_ind % 2 == 0:
                        str_user_prompts += prompt_i
                    else: 
                        str_assistant_prompts += prompt_i
                else:
                    if prompt_ind % 2 == 0:
                        str_user_prompts += prompt_i['content']
                    else:
                        str_assistant_prompts += prompt_i['content']
                
                
                if prompt_ind % 2 == 0:
                    claude_prompt.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": str_user_prompts}],
                                }
                            )
                else:
                    claude_prompt.append(
                                {
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": str_assistant_prompts}],
                                }
                            )
                prompt_ind += 1
            
            str_final_prompt = new_prompt
            claude_prompt.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str_final_prompt}],
                }
            )
        count = -1
        num_calls = 0
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0
        time.sleep(5)
        t.tic()
        while count < 5 and num_calls < 10:
            print('try number: ', num_calls + 1)
            
            print('Prompting LLM with model: ', model, ' and temperature: ', num_calls / 10)
            body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 100,
                        "messages": claude_prompt,
                        "temperature": num_calls / 10,
                    }
        
            response = client.invoke_model(
                modelId=model, body=json.dumps(body)
            )
            
            num_calls += 1
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])
            output = output_list[0]["text"]
            count = len(output)
            response_["generation"] = output
            response_["prompt_token_count"] += input_tokens
            response_["generation_token_count"] += output_tokens
            response = response_
            completion = output
            if len(completion) == 0:
                print('empty completion')
                count = 0
                time.sleep(5)
                
            
            completion, count = self.check_response(completion)
        LLM_time = t.tocvalue()
        return completion, response_["prompt_token_count"], response_["generation_token_count"], LLM_time
    
    def vlm_response(self, prompt, model, graph, all_obs=[]):
        log_dir = self.log_dir
        if 'opus' in model:
            modelid = 'anthropic.claude-3-opus-20240229-v1:0'
            region ="us-west-2"
        else:
            modelid = 'anthropic.claude-3-sonnet-20240229-v1:0'
            region ="us-east-1"
            
        client = boto3.client('bedrock-runtime',region)
        prompt = prompt[:-1]
        str_prompt = ''
        for p in prompt:
            str_prompt += p['content']
        str_prompt += 'The agent locations and their respective goal locations are described below to help you choose the leader and waypoints. '
                    
        agent_states = graph.type_states(0, self.n_agent)
        goals = graph.xg
        
        
        LLM_time = 0
        if self.main_leader > -1 or self.leaders == -1:
            agent_states = graph.type_states(0, self.n_agent)
            if self.main_leader == -1:
                
                
                agent_indices = jnp.arange(self.n_agent)
                leader = -1
            else:
                leader = self.main_leader
                agent_indices = jnp.array([leader])
            image = self.generate_current_grid(graph, all_obs=all_obs, leader=leader)
            response, in_tokens, out_tokens, LLM_time = self.claude_client_vlm(agent_states, goals, image, modelid, client, str_prompt, agent_indices)
            leader_new, waypoints = self.LLM_leader_waypoint_fn(response, graph, self.num_agents)
            if self.main_leader == -1 and leader_new is not None:
                leader = leader_new
                self.main_leader = leader
            
            if waypoints is None:
                c = np.arange(0, 1, 0.2)
                waypoints = np.array([goals[leader] + c[i] * (agent_states[leader] - goals[leader]) for i in range(len(c))])
            if len(waypoints) < self.num_waypoints:
                leader_goal = goals[leader]
                add_waypoints = jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoints), axis=0).squeeze().reshape(-1,2)
                waypoints = jnp.concatenate([waypoints, add_waypoints], axis=0)
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write(str(response))
                f.write('\n')
            waypoints = waypoints[None, ...]
        else:
            in_tokens = 0
            out_tokens = 0
            num_clusters = max(self.assignments) + 1
            waypoints = jnp.zeros((num_clusters, self.num_waypoints, 2))
            LLM_time = 0
            for i in range(num_clusters):
                agent_indices = jnp.where(self.assignments == i)[0]
                agent_states_i = agent_states[agent_indices]
                num_agents = len(agent_indices)
                goals_i = goals[agent_indices]
                leader = agent_indices[self.leaders[i]]
                image = self.generate_current_grid(graph, all_obs=all_obs, leader=leader)
                response, in_token, out_token, LLM_time_cluster = self.claude_client_vlm(agent_states_i, goals_i, image, modelid, client, str_prompt, jnp.array([leader]))
                in_tokens += in_token
                out_tokens += out_token
                LLM_time += LLM_time_cluster
                _, waypoint = self.LLM_leader_waypoint_fn(response, graph, num_agents)
                if len(waypoint) < self.num_waypoints:
                    leader_goal = goals[agent_indices[self.leaders[i]]]
                    waypoint = jnp.concatenate([waypoint, jnp.repeat(leader_goal[None, :], self.num_waypoints - len(waypoint), axis=0)], axis=0)
                else:
                    waypoint = waypoint[:self.num_waypoints]
                waypoints = waypoints.at[i].set(waypoint)
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write(str(response))
                    f.write('\n')
        return waypoints, in_tokens, out_tokens, LLM_time
 
    def claude_client_vlm(self, agent_states, goals, image, modelid, client, str_prompt, agent_indices):
        j = 0
        
        final_prompt = ''
        for i in agent_indices:
            i = agent_indices[0]
        
            x = agent_states[i, 0]
            y = agent_states[i, 1]
            final_prompt += '***AgentId***' + str(j + 1) + '***current state***(' + '{:.1f}'.format(x.item()) + ',' + '{:.1f}'.format(y.item()) + ')'
            final_prompt += '***goal location***(' + '{:.1f}'.format(goals[i, 0].item()) + ',' + '{:.1f}'.format(goals[i, 1].item()) + ')'
            j += 1
        
        print('Last prompt: ', final_prompt)
        
        
        base64_image = encode_image(image)
        attempt = 0  
        response_ = {}
        response_["prompt_token_count"] = 0
        response_["generation_token_count"] = 0

        if self.vlm_example_prompts is not None and self.num_incontext_prompts > 0:
            prompt_0 = self.vlm_example_prompts[0]
            message = [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": str_prompt + prompt_0['content'][0]['text'],
                                },
                            ],
                        }]
            
            message_new = {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please provide the examples and their correct responses.",
                                },
                            ],
                        }
            message.append(message_new)
            vlm_example_prompts = self.vlm_example_prompts[1:]
            j = 0
            for prompt_i in vlm_example_prompts:
                if j % 2 == 0:
                    message_new = {
                                "role": "user",
                                "content": prompt_i['content'],
                            }
                else:
                    message_new = {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": str(prompt_i['content']),
                                    },
                                ],
                            }
                j += 1
                message.append(message_new)
            message_last = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": final_prompt,
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                            ],
                        }
            message.append(message_last)
            
            
            
        else:
            str_prompt = str_prompt + final_prompt
            message = [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": str_prompt,
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                },
                            ],
                        }]
            
            
            
        
        t.tic()
        while attempt < 10:
            try:
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "temperature": attempt / 10,
                    "messages": message,
                }
                response = client.invoke_model(
                    modelId=modelid,
                    body=json.dumps(request_body),)
            except:
                attempt += 1
                print('error in response')
                time.sleep(5)
                continue
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])
            completion, count = self.check_response(output_list[0]["text"])
            if count == 0:
                print('empty completion')
                attempt += 1
            else:
                attempt = 100
            
            response_["generation"] = completion
            response_["prompt_token_count"] += input_tokens
            response_["generation_token_count"] += output_tokens

            response = response_
            return_message = completion
        LLM_time = t.tocvalue()
        print('VLM response: ',response)
        return return_message, response_["prompt_token_count"], response_["generation_token_count"], LLM_time
    
    def get_response(self, prompt, model, LLM_calls=1, json_output=True, graph=None, all_obs=[]):
        """
        Get the response from the LLM model given the prompt.
        Args:
            prompt: the prompt to be sent to the LLM model
            model: the model to be used for the LLM
            LLM_calls: the number of calls to the LLM model
            fixed_prompts: the fixed prompts to be used for the LLM model (problem description and in-context examples)
            new_prompt: the new prompt to be used for the LLM model (current deadlock environment description)
            graph: the graph to be used for the data for LLM model
        Returns:
            return_message: JSON output with leader assignment from the LLM model
            response: the complete response object from the LLM model
        """
        
        LLM_time = 0
        if 'gpt' in model:
            waypoints, in_tokens, out_tokens, LLM_time = self.gpt_response(prompt, model, LLM_calls, all_obs, json_output, graph)
        elif 'llama' in model:
            waypoints, in_tokens, out_tokens, LLM_time = self.llama_response(graph, prompt, model)
        elif 'mistral' in model:
            waypoints, in_tokens, out_tokens, LLM_time = self.mistral_response(graph, prompt, model)
        elif 'claude-v2' in model:
            waypoints, in_tokens, out_tokens,LLM_time = self.claude2_response(graph, prompt, model)    
        elif 'claude-3' in model:
            waypoints, in_tokens, out_tokens, LLM_time= self.claude3_response(graph, prompt, model)
        elif 'vlm' in model:
            waypoints, in_tokens, out_tokens, LLM_time  = self.vlm_response(prompt, model, graph, all_obs)
        else:
            NotImplementedError(f"Model {model} not supported.")
        
        return waypoints, in_tokens, out_tokens, LLM_time

    def get_info_single_graph(self, graph):
        """
        Get the information of a single graph.
        Args:
            graph: the graph to get the information from
            n_agent: the number of agents in the graph
        Returns:
            agent_states: the states of the agents in the graph
            global_goal_states: the global goal states of the agents in the graph
            temp_goal_states: the temporary goal states of the agents in the graph
            connectivity: the connectivity of the agents in the graph
            lidar_data: the lidar data of the agents in the graph
        """
        n_agent = self.n_agent
        agent_states = graph.type_states(0, n_agent)
        global_goal_states = graph.xg
        temp_goal_states = graph.type_states(1, n_agent)
        connectivity = graph.connectivity
        obstacles = graph.env_states.obstacle
        get_lidar_vmap = jax.vmap(
            jax.jit(ft.partial(
                get_lidar,
                obstacles=obstacles,
                num_beams=32,
                sense_range=0.5,
            ))
        )
        lidar_data = get_lidar_vmap(agent_states)
        return agent_states, global_goal_states, temp_goal_states, connectivity, lidar_data

    
    

    def create_user_prompt_fn(self, graph, n_agent, iter, num_waypoints, all_obs=[], use_local_leader=False, use_normalized_data=False):
        """
        Create the user prompt for the LLM model.
        Args:
            graph: the graph to create the user prompt from
            n_agent: the number of agents in the graph
            iter: the iteration number
            use_local_leader: whether to use a local leader
            deadlock_graphs: the deadlock graphs
            use_normalized_data: whether to use normalized data
        Returns:
            message: the user prompt message
            far_agent_indices: the indices of the far agents
        """
        num_agents = n_agent
        
        agent_states = graph.type_states(0, num_agents)
        global_goal_states = graph.xg
        agent_states = agent_states[:, :2]

        
        x = agent_states[:, 0]
        y = agent_states[:, 1]
        if len(all_obs) == 0:
            xm = self.xm
            xM = self.xM
            ym = self.ym
            yM = self.yM
            obs = jnp.array([xm, ym])
            obs = obs.T
            obs_w = xM - xm
            obs_h = yM - ym
        else:
            obs = jnp.array(all_obs)
            obs_w = jnp.ones(len(all_obs)) * 0.1
            obs_h = jnp.ones(len(all_obs)) * 0.1

        
        mean_x = jax.numpy.mean(x)
        mean_y = jax.numpy.mean(y)
        if not use_normalized_data:
            mean_x = 0.0
            mean_y = 0.0

        agent_states = agent_states - jnp.array([mean_x, mean_y])
        obs = obs - jnp.array([mean_x, mean_y])
        global_goal_states = global_goal_states - jnp.array([mean_x, mean_y])

        x = agent_states[:, 0]
        y = agent_states[:, 1]
        
        xg = global_goal_states
        
        agent_goal_dist = jax.numpy.linalg.norm(xg - agent_states, axis=-1)

        
        
        close_agent_mask = np.array([True])
        
        if use_local_leader and close_agent_mask.any():
            leader = jnp.argmin(agent_goal_dist)
            far_agent_indices = jnp.array([leader])            
        else:
            far_agent_indices = jax.numpy.arange(num_agents)

        far_agent_count = len(far_agent_indices)
        
        new_prompt = '***Env:' + str(iter)
        new_prompt += '***Number of agents***' + str(far_agent_count)
        new_prompt +='***Safety radius***0.05***Connectivity radius***0.5***'
        new_prompt += 'Agents'
        j = 0
        far_agent_indices = far_agent_indices.tolist()
        for i in far_agent_indices:
            new_prompt += '***AgentId***' + str(j + 1) + '***current state***(' + '{:.2f}'.format(x[i].item()) + ',' + '{:.2f}'.format(y[i].item()) + ')***goal location***(' + '{:.2f}'.format(xg[i][0].item()) + ',' + '{:.2f}'.format(xg[i][0].item()) + ')' 
            j += 1
            
        new_prompt += '***obstacles***['
        obs_for_prompts = obs
        
        
        if type(obs_for_prompts) == list:
            obs_for_prompts = jnp.array(obs_for_prompts)
        len_obs = len(obs_for_prompts)
        for i in range(len_obs):
            new_prompt += '(x=' + '{:.2f}'.format(obs_for_prompts[i][0].item()) + ', y=' + '{:.2f}'.format(obs_for_prompts[i][1].item())
            new_prompt +=  ', w=' + '{:.2f}'.format(obs_w[i].item()) + ',h=' + '{:.2f}'.format(obs_h[i].item()) + ')'
            if i < len_obs - 1:
                new_prompt += ','

        if new_prompt[-1] == ',':
            new_prompt = new_prompt[:-1]
            
        if new_prompt[-1] != '[':
            new_prompt += ']'
        else:
            new_prompt += 'None]'
        
        new_prompt += '***Number of waypoints***' + str(num_waypoints)

        message = {"role": "user", "content": new_prompt}

        return message, jnp.array(far_agent_indices)

    def find_bounding_box(self, obs, r, xm, xM, ym, yM):
        if len(obs) == 0:
            return xm, xM, ym, yM
        
        if len(xm) == 0:
            obs = np.array(obs)
            x_min = []
            x_max = []
            y_min = []
            y_max = []
            while len(obs) > 0:
                obs_i = obs[0]
                x_min_i = obs_i[0] - r / 2
                x_max_i = obs_i[0] + r / 2
                y_min_i = obs_i[1] - r / 2
                y_max_i = obs_i[1] + r / 2
                
                x_min.append(x_min_i)
                x_max.append(x_max_i)
                y_min.append(y_min_i)
                y_max.append(y_max_i)
                
                obs = obs[((obs[:, 0] < x_min_i - r/ 2) | (obs[:, 0] > x_max_i + r / 2) | (obs[:, 1] < y_min_i - r/ 2) | (obs[:, 1] > y_max_i + r / 2))]
            
            x_min = np.array(x_min).round(2)
            x_max = np.array(x_max).round(2)
            y_min = np.array(y_min).round(2)
            y_max = np.array(y_max).round(2)
        else:
            x_min = np.copy(xm)
            x_max = np.copy(xM)
            y_min = np.copy(ym)
            y_max = np.copy(yM)
            obs = np.array(obs)
            for j in range(len(x_min)):
                
                obsxm = obs[:, 0] >= x_min[j] - r
                obsxM = obs[:, 0] <= x_max[j] + r
                obsym = obs[:, 1] >= y_min[j] - r
                obsyM = obs[:, 1] <= y_max[j] + r
                
                obsxm_s = jnp.abs(obs[:, 0] - x_min[j]) < 0.01
                obsxM_s = jnp.abs(x_max[j] - obs[:, 0]) < 0.01
                obsym_s = jnp.abs(obs[:, 1] - y_min[j]) < 0.01
                obsyM_s = jnp.abs(y_max[j] - obs[:, 1]) < 0.01
                
                obs_horizontal = x_max[j] - x_min[j] >= y_max[j] - y_min[j]
                obs_vertical = x_max[j] - x_min[j] <= y_max[j] - y_min[j]
                xmax_change_ind = np.where(obsxM & (obsym_s | obsyM_s)* obs_horizontal)[0] 
                xmin_change_ind = np.where(obsxm & (obsym_s | obsyM_s)* obs_horizontal)[0] 
                ymin_change_ind = np.where(obsym & (obsxm_s | obsxM_s)* obs_vertical)[0] 
                ymax_change_ind = np.where(obsyM & (obsxm_s | obsxM_s)* obs_vertical)[0] 
                
                if len(obs[xmin_change_ind, 0]) > 0:
                    x_min[j] = min(x_min[j], np.min(obs[xmin_change_ind, 0]))
                if len(obs[xmax_change_ind, 0]) > 0:
                    x_max[j] = max(x_max[j], np.max(obs[xmax_change_ind, 0]))
                if len(obs[ymin_change_ind, 1]) > 0:
                    y_min[j] = min(y_min[j], np.min(obs[ymin_change_ind, 1]))
                if len(obs[ymax_change_ind, 1]) > 0:
                    y_max[j] = max(y_max[j], np.max(obs[ymax_change_ind, 1]))
            
            obs_not_in_any_cluster = []
            
            
            for obs_i in obs:
                
                in_any_cluster = False
                for j in range(len(x_min)):
                    if (obs_i[0] >= x_min[j] - r) and (obs_i[0] <= x_max[j] + r) and (obs_i[1] >= y_min[j] - r) and (obs_i[1] <= y_max[j] + r):
                        in_any_cluster = True
                        break
                
                if not in_any_cluster:
                    obs_not_in_any_cluster.append(obs_i)
            
            x_min1, x_max1, y_min1, y_max1 = self.find_bounding_box(obs_not_in_any_cluster, r, [], [], [], [])
            x_min = np.concatenate([x_min, x_min1])
            x_max = np.concatenate([x_max, x_max1])
            y_min = np.concatenate([y_min, y_min1])
            y_max = np.concatenate([y_max, y_max1])

        return x_min, x_max, y_min, y_max

    def create_assistant_prompt_fn(self,leader_id, dir, graph, n_agent):
        """
        Create the assistant prompt for the LLM model.
        Args:
            leader_id: the id of the leader
            dir: the direction of the leader
            graph: the graph to create the assistant prompt from
            n_agent: the number of agents in the graph
        Returns:
            output_message: the assistant prompt message
        """
        
        
        leader_i = leader_id
        leader_dir_i = dir
        
        lead_rs = leader_i
        str_output = ' {"Leader": ' + str(lead_rs.item() + 1) + ','    
        if graph is None:
            if jnp.linalg.norm(leader_dir_i - jnp.array([1, 0])) < 0.1:
                str_output += '"Direction": "To right"'
            elif jnp.linalg.norm(leader_dir_i - jnp.array([-1, 0])) < 0.1:
                str_output += '"Direction": "To left"'
            else:
                str_output += '"Direction": "To goal"'
        else:
            actual_goal = graph.xg[leader_id]
            state_leader = graph.type_states(0, n_agent)
            state_leader = state_leader[leader_id]
            temp_goal = graph.type_states(1, n_agent)
            temp_goal = temp_goal[leader_id]
            if jnp.linalg.norm(temp_goal - actual_goal) < 0.1:
                str_output += '"Direction": "To goal"'
            else:
                dir = temp_goal - state_leader
                dir_ac = actual_goal - state_leader
                if jnp.cross(dir, dir_ac) > 0:
                    str_output += '"Direction": "To right"'
                else:
                    str_output += '"Direction": "To left"'
        str_output += '}'
        output_message = {"role": "assistant", "content": str_output}
        return output_message


    def create_assistant_prompt_waypoint_fn(self, leader_id, waypoint):
        """
        Create the assistant prompt for the LLM model.
        Args:
            leader_id: the id of the leader
            dir: the direction of the leader
            graph: the graph to create the assistant prompt from
            n_agent: the number of agents in the graph
        Returns:
            output_message: the assistant prompt message
        """
                
        waypoint = waypoint.reshape(-1, 2)

        str_output = '"Waypoints": ['
        for i in range(len(waypoint)):
            str_output += '[' + '{:.2f}'.format(waypoint[i, 0].item()) + ',' + '{:.2f}'.format(waypoint[i, 1].item()) + '],'
        str_output = str_output[:-1]
        str_output += ']'
        str_output += '}'
        output_message = {"role": "assistant", "content": str_output}
        return output_message

    def nominal_leader_dir_fn(self):
        return jnp.array(0), jnp.array([1, 0])

    def LLM_leader_dir_fn(self, response, graph, n_agent):
        """
        Get the leader id and direction from the LLM response.
        Args:
            response: the response from the LLM model
            graph: the graph to get the leader id and direction from
            n_agent: the number of agents in the graph
        Returns:
            leader_id: the id of the leader
            LLM_direction: the direction of the leader
        Returns None, None if there is an error in the LLM response.
        """
        try:
            LLM_response = response
            LLM_response = LLM_response.replace('"', "'")
            LLM_response = LLM_response.replace('\n', '')
            LLM_response = LLM_response.replace('\n', '')
            LLM_response = LLM_response.replace(' ', '')
            LLM_response = LLM_response.replace('"', '')
            
            leader = [int(s) for s in LLM_response if s.isdigit()]
            len_leader = len(leader)
            leader_id = 0
            agent_states = graph.type_states(0, n_agent)
            for i in range(len_leader):
                leader_id += 10 ** (len_leader - i - 1) * leader[i]
            leader_id = int(leader_id) - 1
            if 'left' in LLM_response:
                LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
                
                LLM_direction = jnp.array([-LLM_direction[1], LLM_direction[0]])
            elif 'right' in LLM_response:
                LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
                
                LLM_direction = jnp.array([LLM_direction[1], -LLM_direction[0]])
            elif 'goal' in LLM_response:
                LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
                
            else:
                LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
                
            LLM_direction = LLM_direction / (jnp.linalg.norm(LLM_direction) + 1e-6)
        except:
            print('Error in LLM_leader_dir_fn')
            return None, None
        return jnp.array(leader_id), LLM_direction

    def barebone_message(self, LLM_response):
        """
        Clean the LLM response.
        """
        LLM_response = LLM_response.replace('"', "'")
        LLM_response = LLM_response.replace('\n', '')
        LLM_response = LLM_response.replace('\n', '')
        LLM_response = LLM_response.replace(' ', '')
        LLM_response = LLM_response.replace('"', '')
        return LLM_response

    def find_median(self, messages, graph, n_agent):
        """
        Find the median message from the LLM responses.
        Args:
            messages: the messages from the LLM responses
            graph: the graph to get the leader id and direction from
            n_agent: the number of agents in the graph
        Returns:
            leader_id: the id of the leader from the most frequent message
            LLM_direction: the direction of the leader from the most frequent message
        """
        length = len(messages)
        mess_append = []
        leader_id = None
        LLM_direction = None
        for i in range(length):
            response = self.barebone_message(messages[i])
            mess_append.append(response)
        
        sim_count = jnp.zeros((length, 1))
        for i in range(length):
            sim_count = sim_count.at[i].set(mess_append.count(mess_append[i]))
            if any(sim_count > length / 2):
                break
        sim_count_max = jnp.argmax(sim_count)
        median_message = mess_append[int(sim_count_max)] 

        leader_id, waypoints = self.LLM_leader_waypoint_fn(median_message, graph, n_agent)
        
        leader_id = leader_id.astype(jnp.int32)
        return jnp.array(leader_id), waypoints

    def get_leader_id_dir(self, graph: GraphsTuple):
        """
        Get the id of the leader of the graph, return -1 if no leader is found.
        Also return the moving direction of the leader (only meaningful when leader_id is not -1).
        """
        true_goals = graph.xg
        cur_goals = graph.type_states(type_idx=1, n_type=true_goals.shape[0])
        agent_states = graph.type_states(type_idx=0, n_type=true_goals.shape[0])
        leader_id = jnp.argmax(jnp.linalg.norm(cur_goals - agent_states, axis=-1))
        leader_id = jnp.where(jnp.allclose(cur_goals, true_goals), jnp.array([-1]), leader_id)
        leader_dir = cur_goals[leader_id] - agent_states[leader_id]
        return leader_id, leader_dir
        
    def LLM_leader_waypoint_fn(self, response, graph, n_agent, use_normalized_data=False):

        if len(response) == 0:
            return None, None
        try:
            if '{' in response and '}' in response:
                response = response[response.index('{'):response.index('}')+1]
            json_msg = json.loads(response.replace("\\", ''))
            if type(json_msg) == list:
                json_msg = json_msg[0]
        
            leader_id = json_msg['Leader'] - 1
            waypoints = json_msg['Waypoints']
            waypoints_ = []
            
            if use_normalized_data:
                states = graph.type_states(0, n_agent)
                states = states[:, :2]
                mean_states = jnp.mean(states, axis=0)
            else:
                mean_states = jnp.array([0.0, 0.0])
            
            if not isinstance(waypoints, list):
                for waypoint in waypoints:
                    x = waypoint['x'] + mean_states[0]
                    y = waypoint['y'] + mean_states[1]
                    waypoints_.append(jnp.array([x, y]))
            else:
                waypoints_ = jnp.zeros((len(waypoints), 2))
                i = 0   
                for waypoint in waypoints:
                    x = waypoint[0] + mean_states[0]
                    y = waypoint[1] + mean_states[1]
                    waypoints_ = waypoints_.at[i].set(jnp.array([x, y]))
                    i += 1
            return jnp.array(leader_id), waypoints_
        except:
            return None, None

    def leader_graph(self, graph, jit_policy, jit_leader_follower_assign, kk, leader_control, policy,prompt,all_prompts,reset_graph, leader_assign_count, t_mode, prompt_time_gap, num_LLM_calls, sent_token_count, received_token_count, prev_leader = -1,unsafe_mask_jit=None, envstate_jit=None, getgraph_jit=None,jit_mulit_leader_follower_assign=None):        
        """
        Assign the leader to the agents in the graph.
        Args:
            graph: the graph to assign the leader to
            num_agents: the number of agents in the graph
            jit_policy: the control policy function
            keep_mode: the number of steps to keep the leader
            jit_leader_follower_assign: the leader follower assignment function
            leader_model: the leader model to be used
            kk: the current iteration number
            log_dir: the log directory
            num_runtime_incontext_prompts: the number of runtime in-context prompts
            leader_control: the leader control function
            policy: the policy to be used for the leader control
            prompt: the prompt to be used for the LLM model
            all_prompts: all the prompts for bookkeeping
            LLM_calls: the number of LLM calls
            reset_graph: the function to reset the graph
            leader_assign_count: the number of leader assignments
            t_mode: the mode of the graph
            prompt_time_gap: the time gap between prompts
            num_LLM_calls: the number of LLM calls
            sent_token_count: the number of tokens sent to the LLM model
            received_token_count: the number of tokens received from the LLM model
        Returns:
            graph: the graph with the leader assigned
            num_LLM_calls: the number of LLM calls
            prompt: the prompt to be used for the LLM model
            all_prompts: all the prompts for bookkeeping
            leader_assign_count: the number of leader assignments
            t_mode: the mode of the graph
            prompt_time_gap: the time gap between prompts
            sent_token_count: the number of tokens sent to the LLM model
            received_token_count: the number of tokens received from the LLM model
        """
        new_obs = []
        num_agents = self.num_agents
        leader_model = self.leader_model
        use_local_leader = self.use_local_leader
        use_normalized_data = self.use_normalized_data
        keep_mode = self.keep_mode
            
        log_dir = self.log_dir
        num_runtime_incontext_prompts = self.num_runtime_incontext_prompts
        LLM_calls = self.LLM_calls
        all_obs = self.all_obs        
        num_waypoints = self.num_waypoints
        use_cot = self.use_cot
        
        leader_id = self.leaders

        agent_states = graph.type_states(0, num_agents)
        goals = graph.type_states(1, num_agents)
        if not type(self.waypoints) == int :
            waypoints = self.waypoints.reshape(-1, 2)
        else:
            waypoints = jnp.array([])
            
        if waypoints.shape[0] > 0 and self.main_leader > -1:
            leader_id = self.main_leader
            lead_waypoint_dist = jnp.linalg.norm(waypoints[0] - agent_states[leader_id])
            if lead_waypoint_dist < 0.1:
                if waypoints.shape[0] == 1:
                    waypoints = jnp.array([])
                else:
                    waypoints = waypoints[1:].reshape(-1,2)
                    waypoints = jnp.concatenate([waypoints, goals[leader_id].reshape(1, 2)], axis=0)

        self.all_graphs.append(graph)
        self.iter = kk
        actions = jit_policy(graph)
        action_norm = jnp.linalg.norm(actions, axis=-1)
        avg_speed = jnp.mean(action_norm)
        
        low_speed = avg_speed < 0.2
        far_from_goal = jnp.linalg.norm(graph.xg - agent_states[:, :2], axis=-1).mean() > 0.5

        c_ = jnp.linspace(0, 1, 20)
        c_ = c_[..., None, None]
        points = (1 - c_) * agent_states[None,...] + c_ * graph.xg[None,...]
        
        currect_connectivity = graph.connectivity
        current_goals = graph.xg
        current_obs = graph.env_states.obstacle
        env_state_fn = ft.partial(envstate_jit, obstacle=current_obs, goal=current_goals)

        env_states = jax.vmap(env_state_fn)(points)
        
        graphs_for_points_fn = ft.partial(getgraph_jit, adjacency=currect_connectivity, goals=current_goals)
        graphs_for_points = jax.vmap(graphs_for_points_fn)(env_states)

        
        unsafe_masks = unsafe_mask_jit(graphs_for_points)

        
        unsafe_mode = jnp.any(unsafe_masks, axis=0).any()
        
        use_leader_mode = low_speed & far_from_goal & unsafe_mode

        
        in_leader_mode = ~jnp.allclose(goals, graph.xg)

        leader_call_time = 0
        if (in_leader_mode and prompt_time_gap < keep_mode and len(waypoints) > 0):
            
            waypoints = waypoints.squeeze()        
            graph = jit_mulit_leader_follower_assign(graph, self.leaders, waypoints[0], waypoints, self.assignments, self.main_leader)
            
            prompt_time_gap += 1
        else:
            if use_leader_mode: 
                if num_agents > 10:
                    _, assignments = kmeans(agent_states, num_agents // 10 + 1)
                    if min(assignments) > 0:
                        assignments = assignments - min(assignments)
                    assert max(assignments) <= num_agents // 10
                    
                    num_agents_per_cluster = jnp.bincount(assignments)
                    if jnp.any(num_agents_per_cluster == 1):
                        
                        single_agent_in_clusters = jnp.where(num_agents_per_cluster == 1)[0]
                        agent_distances = jnp.linalg.norm(agent_states[:, None, :2] - agent_states[None, single_agent_in_clusters, :2], axis=-1)
                        j = 0
                        for ind in single_agent_in_clusters:
                            agent_distances = agent_distances.at[ind, j].set(100000)
                            j += 1
                        nearest_cluster = jnp.argmin(agent_distances, axis=0)
                        assignments = assignments.at[single_agent_in_clusters].set(assignments[nearest_cluster])
                    num_assignments = max(assignments) + 1
                    leaders = jnp.zeros((num_assignments,))
                    main_leader = -1
                    # min_goal_dist_global = 10000
                    goal_dist = jnp.linalg.norm(goals[:, :2] - agent_states[:, :2], axis=-1)       
                    main_leader = jnp.argmin(goal_dist)
                    main_leader_cluster = assignments[main_leader]
                    agent_index_main = jnp.where(assignments == main_leader_cluster)[0]
                    for i in range(num_assignments):
                        if i == main_leader_cluster:
                            leaders = leaders.at[i].set(main_leader)
                            continue
                        else:
                            agent_index_i = jnp.where(assignments == i)[0]
                            if len(agent_index_i) == 0:
                                continue
                            agent_dist = jnp.linalg.norm(agent_states[None, agent_index_i, :2] - agent_states[agent_index_main, None, :2], axis=-1)
                            min_dist_id = jnp.argmin(agent_dist)
                            agent_i = min_dist_id % len(agent_index_i)
                            leader_id = agent_index_i[agent_i]
                            leaders = leaders.at[i].set(leader_id)
                    
                    leaders = leaders.astype(jnp.int32)
                    self.leaders = leaders
                    self.main_leader = main_leader
                    # self.leaders = -1
                    # self.assignments = jnp.zeros((num_agents,)).astype(jnp.int32)
                    self.assignments = assignments
                else:
                    self.main_leader = -1
                    self.leaders = -1
                    self.assignments = jnp.zeros((num_agents,)).astype(jnp.int32)
                if leader_model == 'fixed':
                    print('Fixed leader for step: ', kk)
                    leader_id = 0
                    leader_dir = goals[leader_id] - agent_states[leader_id]
                    graph = jit_leader_follower_assign(graph, leader_id, leader_dir + agent_states[leader_id])
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write('Fixed leader at step: ')
                        f.write(str(kk))
                        f.write('\n')
                    num_LLM_calls += 1
                    prompt_time_gap = 0
                elif leader_model == 'random':
                    print('Random leader for step: ', kk)
                    leader_id = np.random.randint(num_agents)
                    self.main_leader = leader_id
                    # leader_dir = goals[leader_id] - agent_states[leader_id]
                    self.leaders = jnp.array([self.main_leader])
                    self.assignments = jnp.zeros((num_agents,)).astype(jnp.int32)
                    key_int = np.random.randint(0, 1000)
                    key = jax.random.PRNGKey(key_int)
                    waypoints = jax.random.uniform(key, (num_waypoints, 2), minval=-5, maxval=5) + agent_states[leader_id]
                    
                    graph = jit_mulit_leader_follower_assign(graph, self.leaders, waypoints[0], waypoints, self.assignments, self.main_leader)
                    # graph = jit_leader_follower_assign(graph, leader_id, leader_dir + agent_states[leader_id])
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write('Random leader at step: ')
                        f.write(str(kk))
                        f.write('\n')
                    num_LLM_calls += 1
                    prompt_time_gap = 0
                else:
                    agent_states_all, _, _, _, lidar_data = jax.vmap(jax.jit(self.get_info_single_graph))(tree_stack(self.all_graphs))
                    obs = lidar_data
                    obs_dist = jnp.linalg.norm(obs - agent_states_all[:, :, None, :], axis=-1)
                    obs_seen = obs_dist < 0.6
                    if len(all_obs) == 0:
                        all_obs = obs[obs_seen, :].round(2).tolist()
                        new_obs = obs[obs_seen, :].round(2)
                    else:
                        
                        all_obs = all_obs + obs[obs_seen, :].round(2).tolist()
                        new_obs = jnp.array(obs[obs_seen, :].tolist()).round(2)
                    
                    all_obs_np_scaled = jnp.unique((20 * jnp.array(all_obs)).round(0), axis=0)
                    # all_obs_np_scaled = (20 * all_obs_np).round(0)
                    if self.use_N_obs > -1:
                        all_obs_x = all_obs_np_scaled[-self.use_N_obs:, 0].tolist()
                        all_obs_y = all_obs_np_scaled[-self.use_N_obs:, 1].tolist()
                    else:
                        all_obs_x = all_obs_np_scaled[:, 0].tolist()
                        all_obs_y = all_obs_np_scaled[:, 1].tolist()
                    
                    new_obs = jnp.unique(new_obs, axis=0)

                    xm, xM, ym, yM = self.find_bounding_box(new_obs, 0.12, self.xm, self.xM, self.ym, self.yM)
                    xm1, xmind = jnp.unique(xm, return_index=True)
                    xM1, xMind = jnp.unique(xM, return_index=True)
                    ym1, ymind = jnp.unique(ym, return_index=True)
                    yM1, yMind = jnp.unique(yM, return_index=True)
                    ind_all = jnp.unique(jnp.concatenate((xmind, xMind, ymind, yMind)))
                    xm = xm[ind_all]
                    xM = xM[ind_all]
                    ym = ym[ind_all]
                    yM = yM[ind_all]

                    self.xm = xm
                    self.xM = xM
                    self.ym = ym
                    self.yM = yM
                    self.all_graphs = []
                    
                    if leader_assign_count < num_runtime_incontext_prompts or leader_model is None: 
                        print('Hand-designed leader for step: ', kk)
                        user_prompt, agent_indices = self.create_user_prompt_fn(graph, num_agents, kk, num_waypoints, use_normalized_data=use_normalized_data, use_local_leader=use_local_leader)

                        goal_dist = jnp.linalg.norm(goals[:, :2] - agent_states[:, :2], axis=-1)
                        leader_id = jnp.array([jnp.argmin(goal_dist)])
                        self.leaders = leader_id
                        self.main_leader = self.leaders
                        if self.use_RRT:
                            samples, sample_flag, leader_call_time = self.find_samples(graph, num_agents, self.main_leader, all_obs)
                        else:
                            samples, sample_flag, leader_call_time = self.find_samples_astar(graph, num_agents, self.main_leader, [], all_obs_x, all_obs_y)
                        len_samples = len(samples)
                        samples = jnp.array(samples).round(2)
                        samples = samples[1:]
                        if len_samples > num_waypoints:
                            samples = samples[::len_samples // num_waypoints]
                        # else:
                        #     samples = samples[:num_waypoints]
                        if len(samples) < num_waypoints:
                            samples = jnp.concatenate((samples, graph.xg[leader_id][None, :].repeat(num_waypoints - len(samples), axis=0).reshape(-1, 2)))
                        assistant_prompt = self.create_assistant_prompt_waypoint_fn(self.main_leader, samples)
                        waypoints = samples
                        
                        if self.incontext_file is not None and sample_flag == 1:
                            with open(self.incontext_file, "a") as f:
                                f.write(str(user_prompt))
                                f.write('\n')
                                f.write(str(assistant_prompt))
                                f.write('\n')
                        
                        leader_assign_count += 1
                        prompt.append(user_prompt)
                        prompt.append(assistant_prompt)
                        prompt_time_gap = 0
                        with open(f"{log_dir}/log_message.txt", "a") as f:
                            f.write('Hand-designed leader at step: ')
                            f.write(str(kk))
                            f.write('\n')
                            f.write(str(user_prompt))
                            f.write('\n')
                            f.write(str(assistant_prompt))
                            f.write('\n')
                        num_LLM_calls += 1
                    else:
                        
                        print('LLM for step: ', kk)
                        user_prompt, agent_indices = self.create_user_prompt_fn(graph, num_agents, kk, num_waypoints, use_normalized_data=use_normalized_data, use_local_leader=use_local_leader)
                        LLM_prompt = np.array(prompt).copy()
                        LLM_prompt = LLM_prompt.tolist()
                        LLM_prompt.append(user_prompt)
                        all_prompts.append(user_prompt)
                        if use_cot:
                            cot_prompt = {"role": "user", "content": 'Provide a step-by-step reasoning for the leader assignment and the waypoints for the leader in the presence of the given obstacles. Give a detailed description of how the leader will be assigned and the waypoints will be chosen. Check these steps for each of the agents and decide which agent can be the leader. Then, choose waypoints for the leader.'}
                            LLM_prompt.append(cot_prompt)
                            all_prompts.append(cot_prompt)
                        
                        
                        with open(f"{log_dir}/log_message.txt", "a") as f:
                            f.write('LLM queried at step: ')
                            f.write(str(kk))
                            f.write('\n')
                            f.write(str(user_prompt))
                            f.write('\n')
                        if len(agent_indices) == 0:
                            print('No agents to assign leader')
                        else:
                            if use_cot:
                                mess_rep, response = self.get_response(LLM_prompt, leader_model, LLM_calls,json_output=False)
                                sent_token_count += response['prompt_token_count']
                                received_token_count += response['generation_token_count']
                                new_message = {'role': 'assistant', 'content': mess_rep}
                                LLM_prompt.append(new_message)
                                all_prompts.append(new_message)
                                new_user_message = {'role': 'user', 'content': 'Based on the reasoning above, please provide the leader id and the waypoints for the leader in the JSON format.'}
                                LLM_prompt.append(new_user_message)
                                all_prompts.append(new_user_message)
                                mess_rep, response = self.get_response(LLM_prompt, leader_model, LLM_calls,json_output=True)
                                print(mess_rep)
                            else:
                                waypoints, in_tokens, out_tokens, leader_call_time = self.get_response(LLM_prompt, leader_model, LLM_calls,json_output=True, graph=graph, all_obs=all_obs)
                                sent_token_count += in_tokens
                                received_token_count += out_tokens
                            num_LLM_calls += 1
                        assistant_prompt = self.create_assistant_prompt_waypoint_fn(self.main_leader, waypoints)
                    
                    waypoints = waypoints.squeeze()     
                    if type(self.leaders) == int:
                        if self.leaders == -1:
                            self.leaders = jnp.array([self.main_leader])
                    elif len(self.leaders) == 0 or (self.leaders==-1).all():
                        self.leaders = jnp.array([self.main_leader])
                    graph = jit_mulit_leader_follower_assign(graph, self.leaders, waypoints[0], waypoints, self.assignments, self.main_leader)
                    
                    
                    all_prompts.append(assistant_prompt)
                    prompt_time_gap = 0
                                
            else:
                
                if in_leader_mode:
                    
                    graph = reset_graph(graph, waypoints)
    
        self.all_obs = all_obs
            
        self.waypoints = waypoints
        return graph, num_LLM_calls, prompt, all_prompts, leader_assign_count, t_mode, prompt_time_gap, sent_token_count, received_token_count, leader_id, leader_call_time
    