import logging
import argparse
from types import SimpleNamespace
from src.enums.enum_logging import EnumLogging
from src.catalogs.component_catalog import ComponentCatalog
from src.utils.util_enum import get_string_values_list

NAME = "name"
REQUIRED = "required"
TYPE = "type"
DEFAULT = "default_value"
DESC = "description"
CHOICES = "choices"
NARGS = "nargs"
ACTION = "action"
FIELDS = [NAME, REQUIRED, TYPE, DEFAULT, DESC, CHOICES, ACTION]


def config_field(name, required, type, default, desc, choices=None, nargs=None, action=None):
    return {
        NAME: name,
        REQUIRED: required,
        TYPE: type,
        DEFAULT: default,
        DESC: desc,
        CHOICES: choices,
        NARGS: nargs,
        ACTION: action,
    }


def create_namespace_from_fields(field_names, elements):
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


def add_arguments_from_namespace(parser: argparse.ArgumentParser, namespace):
    for name in namespace.var_names:
        field = getattr(namespace, name)
        kwargs = {
            "help": field.description,
            "default": field.default_value,
            "required": False,
        }
        if field.action:
            kwargs["action"] = field.action
        else:
            if field.type:
                kwargs["type"] = field.type
            if field.nargs:
                kwargs["nargs"] = field.nargs
            if field.choices:
                kwargs["choices"] = field.choices
        parser.add_argument(f"--{field.name}", **kwargs)
    return parser


def get_launcher_config():
    launcher_fields = [
        config_field(
            name=comp_name,
            required=True,
            type=str,
            default="",
            desc=f"Launcher configuration for {comp_name}",
            choices=ComponentCatalog.get_values_by_comp_name(comp_name),
        )
        for comp_name in ComponentCatalog._component_names
    ] + [
        config_field(
            name="logging",
            type=str,
            required=False,
            choices=get_string_values_list(EnumLogging),
            default=logging.BASIC_FORMAT,
            desc="Logging configuration",
        ),
    ]
    return create_namespace_from_fields(FIELDS, launcher_fields)


class ConfigCatalog:
    _var_cache = None

    _load_config = create_namespace_from_fields(FIELDS, [
        config_field("concurrent_requesters", True, int, 1, ""),
        config_field("load_time", True, float, None, "Total runtime in seconds"),
        config_field(
            "dont_wait_requests_finish",
            False,
            bool,
            False,
            "Wait until all requests finish (true/false)",
            action="store_true"
        ),
        config_field("request_rate_per_requester", True, float, 1, "request/sec"),
        config_field("prompt_size_range", True, int, (10, 100), "Range of prompt sizes (min, max)", nargs=2),
        config_field("decode_size_range", True, int, (10, 100), "Maximum output tokens per response", nargs=2),
        config_field("prompts_per_request", True, int, 1, "Number of prompts per request"),
        config_field("seed", False, int, None, "Random seed for reproducibility"),
        config_field("prompt_gen_sleep_time", False, float, 0.0, "Sleep time between prompt generation"),
        config_field("num_prompt_gen_threads", True, int, 1, "Prompt generation threads"),
    ])

    _experiment_config = create_namespace_from_fields(FIELDS, [
        config_field("model", True, str, None, "Model name or path"),
        config_field("model_name_alias", False, str, None, "Optional alias for the model"),
        config_field("experiment_key", True, str, None, "Experiment identifier"),
        config_field("experiment_group", False, str, None, "Experiment identifier"),

    ])

    _single_prompt_exp_config = create_namespace_from_fields(
        FIELDS, [config_field("repeat_times", True, int, 1, desc="Number of repetitions for a single prompt")]
    )

    _prompt_variation_config = create_namespace_from_fields(FIELDS, [
        config_field("prompt_sizes", True, int, [10, 20], desc="Prompt sizes to test", nargs="+"),
        config_field("max_out_tokens", True, int, 1, desc="Max output tokens per variation"),
    ])

    _load_exp_config = create_namespace_from_fields(FIELDS, [
        config_field("request_rates_per_requester", True, float, [10, 20], desc="Request rates per requester", nargs="+"),
    ])

    _intervaled_load_config = create_namespace_from_fields(FIELDS, [
        config_field("interval_percentage", True, float, 5, desc="Percentage of interval between loads"),
    ])

    _vllm_serve_config = create_namespace_from_fields(FIELDS, [])
    _launcher_config = get_launcher_config()

    _vllm_config = create_namespace_from_fields(FIELDS, [
        config_field("host", True, str, None, "Server host"),
        config_field("port", True, str, None, "Server port"),
        config_field("endpoint", True, str, None, "API endpoint"),
        config_field("ignore_eos", True, bool, True, "Ignore EOS token"),
        config_field("vllm_request_timeout", False, int, None, "Request timeout for vLLM"),
        config_field("vllm_serve_args", False, str, [], "Additional serve arguments", nargs="*"),
    ])

    _dummy_config = create_namespace_from_fields(FIELDS, [
        config_field("infer_sleep_time", True, float, 0, "Sleep time for dummy inference"),
    ])

    _openai_config = create_namespace_from_fields(FIELDS, [
        config_field("api_key", True, str, "", "OpenAI API key"),
    ])

    _path_config = create_namespace_from_fields(FIELDS, [
        config_field("path_to_csv_filename", True, str, "", "Path to input CSV file"),
        config_field("path_to_save_results", True, str, "", "Path to save output results"),
    ])

    _sensitive_values = [
        _openai_config.api_key.name
    ]
