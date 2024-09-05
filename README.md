# LLM-GCBF+ for deadlock resolution

Jax Official Implementation of CoRL submission 344: ``Deadlock resolution in Connected Multi-Robot Systems using Foundation Models''. 

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n gcbfplus python=3.10
conda activate gcbfplus
cd gcbfplus
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

Note that the results reported in the paper are obtained using ```jax==0.4.23``` on NVIDIA RTX A4000 GPU. The performance may vary with different versions of JAX and different hardware.

## Installation

Install GCBF: 

```bash
pip install -e .
```

Install RRT for sampling-based high-level planner: 

```bash
cd rrt/rrt_algorithms
pip install -e .
```

## Run

### High-level planner for deadlock resolution
To run the high-level planner for deadlock resolution in "Maze" environments, use:

```bash
python -u test_with_LLM.py --path logs/SingleIntegrator/gcbf+/model_with_traj/seed0_20240227110346/ -n 10 --epi 20 --obs 100 --preset_reset --preset_scene 'maze' --max-step 5000 --area-size 15 --keep-mode 100 --nojit-rollout --leader_model 'gpt3.5' --num_waypoints 4 --use_local_leader --no-video --use_N_obs -1
```

where the flags are:
- `-n`: number of agents
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--keep-mode`: keep mode for the high-level planner
- `--epi`: number of episodes
- `--leader_model`: leader model for the high-level planner including 
    - 'gpt3.5' : GPT-3.5 LLM
    - 'gpt4' : GPT-4 LLM
    - 'claude2': Claude2 LLM
    - 'claude3': Claude3-Opus LLM
    - 'gpt-4o' : GPT-4o VLM
    - 'gpt4-vlm': GPT-4-Turbo VLM
    - 'vlm': Claude3-Sonnet VLM
    - 'vlm-opus': Claude3-Opus VLM
    - 'hand': RRT
    - 'none': No leader-assignment
- `--num_waypoints`: number of waypoints for the high-level planner
- `--use_N_obs`: number of obstacles to use in the environment. If -1, all the obstacles seen so far by the MRS are used to generate the prompts for high-level planner. Otherwise, the last N obstacles are used.

The area size for ``N=50`` agents is 15, and for ``N=25`` agents is 10. The number of obstacles are determined by the number of agents and area size automatically at the time of the initialization of the environment.

For testing on "Room" environment, use:

```bash
python -u  test_with_LLM.py --path logs/SingleIntegrator/gcbf+/model_with_traj/seed0_20240227110346/ -n 5 --epi 20 --obs 1 --preset_reset --preset_scene 'rand-box' --max-step 3000 --area-size 4.5 --keep-mode 100 --nojit-rollout --leader_model 'gpt3.5' --num_waypoints 4 --use_local_leader --no-video --use_N_obs -1
```

### Pre-trained models

We provide one pre-trained GCFB+ model in the folder [`logs`](logs).

The details for training the low-level controller can be found at the original repository of GCBF+ [GCBF+ git](https://github.com/MIT-REALM/gcbfplus/).


## Accessign LLMs and VLMs for the high-level planner

### Accessing GPT Models

The code requires access to GPT models via OpenAI API. To access the models, you need to have an OpenAI account and an API key. You can sign up for an account [here](https://platform.openai.com/signup). Once you have an account, you can find your API key [here](https://platform.openai.com/account/api-keys). The instructions to using the API key can be found [here](https://platform.openai.com/docs/api-reference/authentication).

### Accessing Claude Models

The code uses AWS-based access to Claude models through ```boto3``` package. You need to have an AWS account and configure your credentials as described [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).