import os
import yaml

PROJECT_BASE = os.getenv("CPB_PROJECT_BASE") or os.getenv("CPB_DEPLOY_BASE")


def load_base_config(conf_path):
    """Load configuration from a YAML file."""
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
        print(conf_path)
    with open(conf_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def get_project_base_directory(*args):
    global PROJECT_BASE
    # print(PROJECT_BASE)
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
            )
        )

    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE

if __name__ == '__main__':
    print(get_project_base_directory())