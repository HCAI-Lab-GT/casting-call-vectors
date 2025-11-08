# CLASSES

each thing should have a load/unload method
you should be able to save/load both objects: (load from json, save to json)

PersonaDataset (model_id)
    - trait
    - n (num questions)
    - gen_model => BACKEND should be inference-only
        - ollama
        - gtri func 
        
    the exact datastructure you have now

ModelWithPersona (target model_id, Dataset object | None, layer=) (optional dataset becuase the ability to run without steering)
    - init:
        - extract activations, to get the vector
        - backend currently but faster
        - doesPersonaExist = True
        (this is everything that's saved)
    - Inference (prompt, alpha, temp, top_p = 0.9)
        - automatically inferences with persona vector


=> please use types when possible
=> force models to do as little formatting as possible (look at my code for the subtle-diffing) this is brittle and can confound