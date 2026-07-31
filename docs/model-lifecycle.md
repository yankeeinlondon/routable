# Model Lifecycle

## Lifecycle Stages

To understand how the LLM Router works we must understand the _lifecycle_ of an LLM model. We can think of a locally downloaded model as existing in one of the following states:

1. **Downloaded**

    The model has been downloaded (probably from hugging face) but currently it's just a file or set of files on the local filesystem.

    - in configuration we should have a designated folder for model files
        - probably one per runner (though possibly that could be limiting in the future; I think for now it's an ok idea)
    - by being explicit about _where_ model files goes we can react and report on models which have been placed in a directory but _have not_ been ingested into the system so that they can be used


2. **Available**

    For a model to become "available" it must be _ingested_ into the system. Ingestion must be defined in greater detail but would include things like:

    - metadata procurement
        - VRAM size
        - default settings:
            - temperature
            - top_p
            - context window size
            - thinking levels
            - tokenizer
            - draft model (where appropriate)
            - etc.
    - compilation of model (if required)
    - test load, sleep, unload cycle for validation purposes
        - we probably should do this 3-4 times and measure the timing for the given model moving between these various states
        - that metadata could then be used when callers ask for a model which is not yet available, we can then start the activation process but also provide a realistic estimate of how long the activation will take

3. **Needed**

    - Available models move into the **Needed** state when:
        - a caller calls an endpoint using that model's id as reference
        - the router has been configured with a "default model" or has some business logic that makes it decide that an available model should be made active 
    - this should be an explicit state and 

4. **Active**

    A model is **active** when:

    - it has been fully loaded up into the compute resources (CPU and GPU) 
    - and is actively serving clients

5. **Unused**

    The _unused_ state indicates a model is in the highest state of readiness _to be used_ but currently is not being actively used.

    A model transitions to **unused** when:

    - no user requests over a given time interval (defined in configuration, enforced by router) has been breached
    - a user request to "wake up" is received (via the router's API) but no actual request to use the model have been received
    - when a user has _explicitly_ stated that a model should "go to sleep" the router will then determine whether to keep it in the **unused** state or graduate it to the **Archived** state

6. **Archived**

    An _archived_ model is one that no longer resides in the system. While large model files should be removed periodically over time, the router will retain a record of it's settings for reporting pursposes if nothing else. 

    > Note: unlike the weights of a model, the metadata describing it is tiny and keeping it has effectively zero cost

    A model transitions to **archived** when:

    - a user requests the model's removal via the API, or
    - the router detects that the underlying model file has been removed

## Lifecycle Processes

### Ingestion and Archival Processes


```mermaid
flowchart LR 
    
```



### Model Management Process
