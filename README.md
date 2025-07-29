# LLM Performance Benchmark

## Setup
```bash
pip install -r requirements.txt
```

## Running Examples
Examples scripts are in the `examples` folder.

### Single Prompt Experiment
#### vLLM
1) Run the server: 
```
vllm serve facebook/opt-125m
```

2) Wait for it to start, then run:

```
script=examples/single_prompt_exp/run_vllm_opt_125m_example.sh && chmod +x $script && ./$script
```

#### OpenAI
Replace <OPEN_AI_KEY> with your API key and execute the command below:
```
script=examples/single_prompt_exp/run_openai_gpt_3.5_turbo_0125_example.sh && chmod +x $script && ./$script <OPEN_AI_KEY>
```

#### Generating Single Prompt Report
```
script=examples/single_prompt_exp/gen_single_prompt_examples_report.sh && chmod +x $script && ./$script
```

### Load Experiment
#### vLLM
1) Run the server: 
```
vllm serve facebook/opt-125m
```

2) Wait for it to start, then run:

```
script=examples/load_exp/run_vllm_opt_125m_example.sh && chmod +x $script && ./$script
```

#### OpenAI
Replace <OPEN_AI_KEY> with your API key and execute the command below:
```
script=examples/load_exp/run_openai_gpt_3.5_turbo_0125_example.sh && chmod +x $script && ./$script <OPEN_AI_KEY>
```

#### Generating Load Report
If you ran the OpenAI experiment, make sure to add the experiment key in `gen_load_examples_report.sh`, then run:

```
script=examples/load_exp/gen_load_examples_report.sh && chmod +x $script && ./$script
```

### Intervaled Load Experiment
#### vLLM
1) Run the server: 
```
vllm serve facebook/opt-125m
```

2) Wait for it to start, then run:

```
script=examples/intervaled_load_exp/run_vllm_opt_125m_example.sh && chmod +x $script && ./$script
```

#### OpenAI
Replace <OPEN_AI_KEY> with your API key and execute the command below:
```
script=examples/intervaled_load_exp/run_openai_gpt_3.5_turbo_0125_example.sh && chmod +x $script && ./$script <OPEN_AI_KEY>
```

#### Generating Load Report
If you ran the OpenAI experiment, make sure to add the experiment key in `gen_intervaled_load_examples_report.sh`, then run:

```
script=examples/intervaled_load_exp/gen_intervaled_load_examples_report.sh && chmod +x $script && ./$script
```