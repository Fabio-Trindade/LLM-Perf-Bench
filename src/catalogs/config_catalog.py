import logging
from types import SimpleNamespace
from src.enums.enum_logging import EnumLogging
from src.catalogs.component_catalog import ComponentCatalog
from src.registries.component_registry import ComponentRegistry
from src.utils.util_enum import get_string_values_list

NAME = "name"
REQUIRED = "required"
TYPE = "type"
DEFAULT = "default_value"
DESC = "description"
CHOICES = "choices"
NARGS = "nargs"

FIELDS = [NAME, REQUIRED, TYPE, DEFAULT, DESC, CHOICES]

def config_field(name, required, type, default, desc, choices=None, nargs = None):
    return {
        NAME: name,
        REQUIRED: required,
        TYPE: type,
        DEFAULT: default,
        DESC: desc,
        CHOICES: choices,
        NARGS: nargs
    }

def create_namespace_from_fields( field_names, elements):
    sub_namespaces = []
    name_values = []

    for element in elements:
        if isinstance(element, (list, tuple)):
            element = dict(zip(field_names, element))
        name = element[NAME]
        name_values.append(name)
        sub_namespaces.append(SimpleNamespace(**element))

    namespace = SimpleNamespace(**{n: ns for n, ns in zip(name_values, sub_namespaces)})
    setattr(namespace, "var_names", name_values)
    return namespace

def get_launcher_config():
        launcher_fields = [
            config_field(
                name=comp_name,
                required=True,
                type=str,
                default="",
                desc=f"Launcher configuration for {comp_name}",
                choices=ComponentCatalog.get_values_by_comp_name(comp_name)
            )
            for comp_name in ComponentCatalog._component_names
        ] + [
            config_field( name = "logging",  type = str, required = False, choices = get_string_values_list(EnumLogging),
                        default  = logging.BASIC_FORMAT, desc= "Fill later"),
            

        ]
        return create_namespace_from_fields(FIELDS, launcher_fields)

class ConfigCatalog:
    _var_cache = None

    
    _load_config = create_namespace_from_fields(FIELDS, [
        config_field("num_prompts", True, int, 0, "Number of prompts to generate"),
        config_field("prompt_size_range", True, tuple, (10, 100), "Range of prompt sizes (min, max)"),
        config_field("max_out_tokens", True, int, 256, "Maximum output tokens per response"),
        config_field("prompts_per_request", True, int, 1, "Number of prompts per request"),
        config_field("seed", False, int, None, "Random seed for reproducibility"),
        config_field("run_time", True, float, 60.0, "Total runtime in seconds"),
        config_field("prompt_gen_sleep_time", False, float, 0.0, "Sleep time between prompt generation"),
        config_field("requester_sleep_time", False, float, 0.0, "Sleep time between requests"),
        config_field("num_prompt_gen_threads", True, int, 1, "Prompt generation threads"),
        config_field("num_requester_threads", True, int, 1, "Requester threads"),
    ])


    _experiment_config = create_namespace_from_fields(FIELDS, [
        config_field("model", True, str, None, "Model name or path"),
        config_field("model_name_alias", False, str, None, "Optional alias for the model"),
        config_field("experiment_key", True, str, None, "Experiment identifier"),
    ])

    _single_prompt_exp_config = create_namespace_from_fields(FIELDS, [
        config_field("prompt_sizes", True, int, [10,20], desc = "Later", nargs='+'),
        # config_field("max_out_tokens_sizes", True, int, [10,20], desc = "Later", nargs='+')

    ])

    
    #components
    _launcher_config = get_launcher_config()

    _vllm_config = create_namespace_from_fields(FIELDS, [
        config_field("host", True, str,None, "Server host"),
        config_field("port", True, str,None, "Server port"),
        config_field("endpoint", True, str,None, "API endpoint"),
        config_field("vllm_server_init_timeout", True, int,None, "API endpoint"),
        config_field("num_answers", True, int, None, "API endpoint"),
        config_field("use_beam_search", True, bool,None, "API endpoint"),
        config_field("temperature", True, float,None, "API endpoint"),
        config_field("vllm_request_timeout", True , int, None, "API endpoint"),
    ])

    _dummy_config = create_namespace_from_fields(FIELDS, [
        config_field("infer_sleep_time", True, int, 0, "Sleep time for inference"),
    ])

    _openai_config = create_namespace_from_fields(FIELDS, [
        config_field("api_key", True, str, "", "OpenAI API key"),
    ])

    _path_config = create_namespace_from_fields(FIELDS,[
         config_field("path_to_csv_filename",True, str, "", ""),
         config_field("path_to_save_results", True, str, "", "")
    ])