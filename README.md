# Persona Vectors HCAI Strategy Games

For this repo, we have defined the persona vectors for the strategy games research project.

To run the desired behaviors designed in the spec, follow the system below.

Anything involving using the gtri openai llm endpoint, set the local_model to be False in that scenario so that it does not default to mlx.

Generate Trait Dataset: This will generate a traits_output json file for that trait if it does not already exist.

python3 TraitGenerator.py

Generate Persona Vector: One you generate the trait dataset for the trait you desire to embody, these next commands will allow you to extract the persona vector accordingly.

# make sure you set the trait variable to the desired trait you want the persona for, and again set local_model to False if you'd like to use GTRI model.
python3 ActivtionExtractor.py

Inference with Persona: This will take a sample query that you can define and run the inference applying the persona you choose.

# make sure you set the appropriate trait and set the question you desire to be answered.
python3 PersonaInference.py


