class ComponentCatalog:
    _requester_instances = ["dummy","vllm", "openai"]
    _config_instances = ["dummy", "openai"]
    _tokenizer_instances = ["HF", "openai", "whitespace"]
    _dataset_gen_instances = ["synthetic", "replay"]
    
    _types = ["parser", "workload"]
    
    _component_names = ["requester", "tokenizer", "server", "dataset_gen"]

    _comp_instances = [
        _requester_instances,
        _tokenizer_instances,
        _config_instances,
        _dataset_gen_instances
    ] 

    _comp_to_values = {comp_name : comp_values for comp_name,comp_values in zip(_component_names,_comp_instances)} 
    @classmethod
    def get_typenames(cls):
        return cls._types
    
    @classmethod
    def get_comp_names(cls):
        return cls._component_names
    
    @classmethod
    def get_values_by_comp_name(cls, comp_name: str):
        return cls._comp_to_values[comp_name]
    

