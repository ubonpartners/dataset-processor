from src.dataset_processor import DatasetProcessor
from src.dataset_processor import dataset_merge_and_delete
from src.llm_attributes import llm_test
from dataclasses import dataclass
from src.pipeline_stages import (
    build_task_add_attributes,
    build_task_add_faces,
    build_task_add_fiqa,
    build_task_add_objects,
    build_task_add_pose,
    build_task_dedup,
    build_task_export_dataset,
    build_task_filter_classes,
    build_task_generate_backgrounds,
    build_task_import,
    build_task_llm_filter,
    build_task_make_hard,
    build_task_mask_objects,
    build_task_normalise,
    build_task_sample,
)
import src.dataset_util as dsu
import os
import argparse
import time
import copy
from tqdm import tqdm
import stuff


@dataclass(frozen=True)
class StageSpec:
    name: str
    fn: object
    pass_test: bool = False
    list_config: bool = False


TASK_PREP_STAGES = (
    StageSpec("llm_filter", build_task_llm_filter, pass_test=True),
    StageSpec("make_hard", build_task_make_hard, pass_test=True),
    StageSpec("sample", build_task_sample, pass_test=True),
)

CHUNK_STAGES = (
    StageSpec("add_objects", build_task_add_objects, list_config=True),
    StageSpec("mask_objects", build_task_mask_objects, list_config=True),
    StageSpec("add_pose", build_task_add_pose, list_config=True),
    StageSpec("add_faces", build_task_add_faces, list_config=True),
    StageSpec("add_fiqa", build_task_add_fiqa),
)

KNOWN_DATASET_KEYS = {
    "general",
    "import",
    "merge",
    "llm_filter",
    "make_hard",
    "sample",
    "add_objects",
    "mask_objects",
    "add_pose",
    "add_faces",
    "add_fiqa",
    "add_attributes",
    "add_backgrounds",
    "export",
    "filter_classes",
}

LEGACY_DATASET_KEYS = {
    "expand_attributes": "expanded attributes are no longer supported; generated datasets are split v10",
    "write_v10": "generated datasets are already split v10; use export to copy/rename a dataset",
}

DEFAULT_CLASS_NAMES = ["person", "face", "vehicle", "animal", "weapon"]
MODEL_KEYS = ("model", "check_model")
MODEL_EXTENSIONS = (".pt", ".pth", ".onnx", ".engine")

KNOWN_GENERAL_KEYS = {
    "tasks",
    "inherit_config",
    "class_names",
    "attributes",
    "face_kp",
    "pose_kp",
    "chunk_size",
    "generate",
}



def is_path_like_model(model):
    if not isinstance(model, str):
        return False
    return os.path.sep in model or model.endswith(MODEL_EXTENSIONS)

def validate_model_path(dataset_name, stage_name, key, model):
    if model is None or not is_path_like_model(model):
        return
    if not os.path.isfile(model):
        raise ValueError(f"{dataset_name}: {stage_name}.{key} model path does not exist: {model}")

def validate_stage_model_paths(dataset_name, dataset_config):
    for stage_name, config in dataset_config.items():
        for stage_config in iter_stage_configs(config, list_config=isinstance(config, list)):
            if not isinstance(stage_config, dict):
                continue
            for key in MODEL_KEYS:
                validate_model_path(dataset_name, stage_name, key, stage_config.get(key))

def validate_dataset_config(dataset_name, dataset_config):
    legacy_keys=set(dataset_config) & set(LEGACY_DATASET_KEYS)
    if legacy_keys:
        messages=[f"{key}: {LEGACY_DATASET_KEYS[key]}" for key in sorted(legacy_keys)]
        raise ValueError(f"{dataset_name}: legacy dataset config keys are not supported ({'; '.join(messages)})")

    unknown_keys=set(dataset_config)-KNOWN_DATASET_KEYS
    if unknown_keys:
        raise ValueError(f"{dataset_name}: unknown dataset config keys {sorted(unknown_keys)}")

    general_config=dataset_config.get("general")
    if not isinstance(general_config, dict):
        raise ValueError(f"{dataset_name}: missing general config")

    unknown_general_keys=set(general_config)-KNOWN_GENERAL_KEYS
    if unknown_general_keys:
        raise ValueError(f"{dataset_name}: unknown general config keys {sorted(unknown_general_keys)}")

    class_names=general_config.get("class_names")
    if not isinstance(class_names, list) or len(class_names)==0:
        raise ValueError(f"{dataset_name}: class_names must be a non-empty list")
    if any(not isinstance(c, str) or len(c)==0 for c in class_names):
        raise ValueError(f"{dataset_name}: class_names must contain non-empty strings")
    if len(set(class_names))!=len(class_names):
        raise ValueError(f"{dataset_name}: class_names must not contain duplicates")
    if "person" not in class_names:
        raise ValueError(f"{dataset_name}: class_names must include 'person'")

    attributes=general_config.get("attributes")
    if isinstance(attributes, list) and len(attributes)>dsu.ATTR_NC_V10:
        raise ValueError(f"{dataset_name}: split v10 supports at most {dsu.ATTR_NC_V10} attributes")

    if "merge" not in dataset_config and "import" not in dataset_config:
        raise ValueError(f"{dataset_name}: non-merge datasets require an import config")

    for stage_name in ("add_objects", "mask_objects", "add_pose", "add_faces"):
        configs=dataset_config.get(stage_name)
        if configs is None:
            continue
        if not isinstance(configs, list):
            raise ValueError(f"{dataset_name}: {stage_name} must be a list")
        for config in configs:
            per_class_thr=config.get("per_class_thr") if isinstance(config, dict) else None
            if per_class_thr is not None and len(per_class_thr)!=len(class_names):
                raise ValueError(f"{dataset_name}: {stage_name}.per_class_thr must have {len(class_names)} entries")

    validate_stage_model_paths(dataset_name, dataset_config)

def normalise_attribute_name(name):
    if not isinstance(name, str):
        return name
    return name.replace(":", "_")

def apply_filter_class_attribute_deletes(dataset_name, dataset_config):
    filter_config=dataset_config.get("filter_classes")
    if filter_config is None:
        return
    if not isinstance(filter_config, dict):
        raise ValueError(f"{dataset_name}: filter_classes must be a dict")

    attributes_to_delete=filter_config.get("attributes_to_delete")
    if attributes_to_delete is None:
        return
    if not isinstance(attributes_to_delete, list):
        raise ValueError(f"{dataset_name}: filter_classes.attributes_to_delete must be a list")
    if any(not isinstance(a, str) for a in attributes_to_delete):
        raise ValueError(f"{dataset_name}: filter_classes.attributes_to_delete must contain strings")

    general_config=dataset_config.get("general")
    attributes=general_config.get("attributes") if isinstance(general_config, dict) else None
    if not isinstance(attributes, list):
        raise ValueError(f"{dataset_name}: filter_classes.attributes_to_delete requires general.attributes list")

    attr_lookup={}
    for i, attr in enumerate(attributes):
        key=normalise_attribute_name(attr)
        if key not in attr_lookup:
            attr_lookup[key]=[]
        attr_lookup[key].append(i)

    missing=[]
    delete_indices=set()
    for attr in attributes_to_delete:
        key=normalise_attribute_name(attr)
        if key not in attr_lookup:
            missing.append(attr)
            continue
        delete_indices.update(attr_lookup[key])
    if missing:
        raise ValueError(f"{dataset_name}: filter_classes.attributes_to_delete contains unknown attributes {sorted(missing)}")

    if not delete_indices:
        return
    general_config["attributes"]=[a for i, a in enumerate(attributes) if i not in delete_indices]

def iter_stage_configs(config, list_config=False):
    if config is None:
        return []
    if list_config:
        return config
    return [config]

def run_configured_stage(processor, dataset_config, stage_name, stage_fn, *, test=False, list_config=False):
    if stage_name not in dataset_config:
        return
    for stage_config in iter_stage_configs(dataset_config[stage_name], list_config=list_config):
        if test:
            stage_fn(processor, stage_config, test=test)
        else:
            stage_fn(processor, stage_config)

def run_task_prep_stages(processor, dataset_config, test=False):
    for stage in TASK_PREP_STAGES:
        run_configured_stage(
            processor,
            dataset_config,
            stage.name,
            stage.fn,
            test=test if stage.pass_test else False,
            list_config=stage.list_config,
        )

def run_chunk_stages(processor, dataset_config):
    for stage in CHUNK_STAGES:
        run_configured_stage(
            processor,
            dataset_config,
            stage.name,
            stage.fn,
            list_config=stage.list_config,
        )
    build_task_normalise(processor)

    if "add_attributes" in dataset_config:
        add_attributes_config=dict(dataset_config["add_attributes"] or {})
        if add_attributes_config.get("loader") is None:
            add_attributes_config["loader"]=dataset_config["import"]["loader"]
        build_task_add_attributes(processor, add_attributes_config)

def copy_backgrounds_into_dataset(bg_yaml, yaml_path, tasks, class_names):
    for task in tasks:
        processor=DatasetProcessor(bg_yaml, task=task, append_log=False, class_names=class_names)
        dest_path=dsu.get_dataset_path(yaml_path)+"/"+task
        desc="BG"+"/"+task+"/copying"
        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), smoothing=0.01):
            src_img=processor.get_img_path(i)
            src_label=processor.get_label_path(i)
            dst_img=dest_path+"/images/"+os.path.split(src_img)[1]
            dst_label=dest_path+"/labels/"+os.path.split(src_label)[1]
            stuff.rename(src_img, dst_img)
            stuff.rename(src_label, dst_label)

def run_post_dataset_stages(yaml_path, dataset_config, *, class_names, face_kp, pose_kp, tasks, test, dataset_name):
    if "add_backgrounds" in dataset_config:
        bg_config=dataset_config["add_backgrounds"]
        if bg_config is not None and bg_config.get("loader") not in (None, "None"):
            bg_yaml=build_task_generate_backgrounds(class_names,
                                                    bg_config,
                                                    face_kp=face_kp,
                                                    pose_kp=pose_kp,
                                                    test=test,
                                                    config_name=dataset_name)
            copy_backgrounds_into_dataset(bg_yaml, yaml_path, tasks, class_names)
            path=dsu.get_dataset_path(bg_yaml)
            print(f"generate_backgrounds: delete folder {path}...")
            stuff.rmdir(path)

    if "export" in dataset_config:
        export_config=dataset_config["export"]
        yaml_path=build_task_export_dataset(yaml_path,
                                            config=export_config,
                                            face_kp=face_kp,
                                            pose_kp=pose_kp)

    if "filter_classes" in dataset_config:
        filter_config=dataset_config["filter_classes"]
        yaml_path=build_task_filter_classes(yaml_path,
                                            filter_config,
                                            face_kp=face_kp,
                                            pose_kp=pose_kp)
    return yaml_path

def merge(a: dict, b: dict, path=None):
    """
    Recursively merge two dictionaries
    Values in b take precedence, b is merged 'over' a
    """
    if path is None:
        path=[]
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def get_expanded_config(path, config_filename, dataset_name):
    """
    Recursively expand configuration
    'inherit_config' lets one specify a configuration in the same or
    another file to set default options
    """
    this_config = stuff.load_dictionary(os.path.join(path, config_filename))
    dataset_config=this_config["datasets"][dataset_name]
    if "general" in dataset_config:
        if "inherit_config" in dataset_config["general"]:
            inherit_from=dataset_config["general"]["inherit_config"]
            if "/" in inherit_from:
                ih_config_filename, ih_dataset_name=inherit_from.split("/")
            else:
                ih_config_filename, ih_dataset_name=config_filename, inherit_from
            ih_config=get_expanded_config(path, ih_config_filename, ih_dataset_name)
            ih_config=copy.deepcopy(ih_config)
            merge(ih_config, dataset_config)
            return ih_config
    return dataset_config

def collect_model_folders_for_sync(config_path):
    config=stuff.load_dictionary(config_path)
    path=os.path.split(config_path)[0]
    config_filename=os.path.split(config_path)[1]
    folders=set()

    def collect_from_obj(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in MODEL_KEYS and is_path_like_model(value):
                    if value.startswith("/mldata/models/"):
                        folders.add(os.path.dirname(value))
                collect_from_obj(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_from_obj(item)

    for dataset_name in config.get("datasets", {}):
        dataset_config=get_expanded_config(path, config_filename, dataset_name)
        collect_from_obj(dataset_config)

    return sorted(folders)

def generate_dataset(path, config_filename, dataset_name, force_generate=False, test=False):
    config = stuff.load_dictionary(os.path.join(path, config_filename))

    # create new empty dataset
    dataset_config=get_expanded_config(path, config_filename, dataset_name)
    apply_filter_class_attribute_deletes(dataset_name, dataset_config)
    # check for OPENAI_API_KEY so it doesn't annoyingly crash later
    if "add_attributes" in dataset_config:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            print("Error: OPENAI_API_KEY is not set")
            exit()

    general_config=dataset_config["general"]
    class_names=general_config["class_names"]
    attributes=general_config.get("attributes", None)
    face_kp=general_config.get("face_kp", False)
    pose_kp=general_config.get("pose_kp", False)
    tasks=["val","train"]
    chunk_size=10000
    if "generate" in general_config:
        if general_config["generate"] is False and force_generate is False:
            return None

    validate_dataset_config(dataset_name, dataset_config)

    if "tasks" in general_config:
        tasks=general_config["tasks"]

    if "chunk_size" in general_config:
        chunk_size=general_config["chunk_size"]

    unique_dataset_name=dsu.unique_dataset_name("output_datasets/"+dataset_name)

    yaml_path=None

    if "merge" in dataset_config:
        datasets_to_merge=dataset_config["merge"]
        for d in datasets_to_merge:
            to_merge_yaml=generate_dataset(path, config_filename, d, force_generate=True, test=test)
            processor=DatasetProcessor(to_merge_yaml)

            if yaml_path is None:
                # no classes are passed, this DP will be configured according to the yaml file
                # then we use it's values in creating our output yaml
                # long winded way of making sure the merged yaml has the same properties as the
                # original ones
                processor=DatasetProcessor(to_merge_yaml)
                merged_config_class_names=processor.class_names
                merged_config_attributes=processor.attributes
                merged_config_face_kp=processor.face_kp
                merged_config_pose_kp=processor.pose_kp
                yaml_path=dsu.make_dataset_yaml(unique_dataset_name,
                                    config_name=dataset_name,
                                    class_names=merged_config_class_names,
                                    face_kp=merged_config_face_kp,
                                    pose_kp=merged_config_pose_kp,
                                    attributes=merged_config_attributes)
            assert processor.class_names==merged_config_class_names
            assert processor.attributes==merged_config_attributes
            assert processor.face_kp==merged_config_face_kp
            assert processor.pose_kp==merged_config_pose_kp

            dataset_merge_and_delete(to_merge_yaml, yaml_path)

        for task in ["val","train"]:
            x=DatasetProcessor(yaml_path, task=task, append_log=True)
            build_task_dedup(x)
            x.log("======================")
            x.log(" Final merged stats: "+str(x.basic_stats()))
        return yaml_path

    yaml_path=dsu.make_dataset_yaml(unique_dataset_name,
                                    config_name=dataset_name,
                                    class_names=class_names,
                                    face_kp=face_kp,
                                    pose_kp=pose_kp,
                                    attributes=attributes)
    # Overall wall-clock timer (across both val/train tasks, backgrounds, etc).
    overall_start_time=time.time()
    for task in tasks:
        processor=DatasetProcessor(yaml_path,
                                   task=task,
                                   append_log=True,
                                   face_kp=face_kp,
                                   pose_kp=pose_kp)

        import_config=dataset_config["import"]
        build_task_import(processor, import_config, test=test)
        run_task_prep_stages(processor, dataset_config, test=test)

        # Per-task timer used for chunk progress estimation.
        start_time=time.time()
        processor.chunk_size=chunk_size
        processor.reload_files()

        for chunk in range(processor.num_chunks):
            print(f"\n==== {processor.config_name} {processor.task} : chunk {chunk} of {processor.num_chunks} ====")
            t=time.time()
            processor.set_chunk(chunk)

            run_chunk_stages(processor, dataset_config)

            elapsed=time.time()-t
            total_elapsed=time.time()-start_time
            remaining=(processor.num_chunks-chunk-1)*total_elapsed/(chunk+1.0)

            print(f"chunk took {dsu.timestr(elapsed)};"
                  +f" total {dsu.timestr(total_elapsed)};"
                  +f" remaining {dsu.timestr(remaining)}")
    print(f"\n==== {processor.config_name} {processor.task} : all chunks complete ====\n")

    yaml_path=run_post_dataset_stages(
        yaml_path,
        dataset_config,
        class_names=class_names,
        face_kp=face_kp,
        pose_kp=pose_kp,
        tasks=tasks,
        test=test,
        dataset_name=dataset_name,
    )

    total_elapsed=int(time.time()-overall_start_time)
    for task in ["val", "train"]:
        x=DatasetProcessor(yaml_path, task=task, append_log=True)
        x.log(" Final merged stats: "+str(x.basic_stats()))
        if task=="train":
            x.log("======================")
            x.log(f" Total elapsed time {dsu.timestr(total_elapsed)}")
    return yaml_path

def postprocess_datasets(config):
    pp_config=config["postprocess"]
    in_datasets=[]
    out_datasets=[]
    if "input_folder" in pp_config:
        input_folder=pp_config["input_folder"]
        if "output_folder" in pp_config:
            output_folder=pp_config["output_folder"]
        else:
            output_folder=input_folder
        output_folder=dsu.unique_dataset_name(output_folder)

        folders=os.listdir(dsu.mldata_folder()+"/"+input_folder)
        for f in folders:
            in_datasets.append(input_folder+"/"+f)
            out_datasets.append(output_folder+"/"+f)
    for i,_ in enumerate(in_datasets):
        build_task_filter_classes(dsu.mldata_folder()+"/"+in_datasets[i]+"/dataset.yaml",
                                  config=pp_config,
                                  output_dataset_name=out_datasets[i])




def warn_if_no_final_merge_target(config):
    datasets=config.get("datasets", {})
    generated_merges=[]
    generated=[]
    for dataset_name, dataset_config in datasets.items():
        general=dataset_config.get("general", {}) if isinstance(dataset_config, dict) else {}
        if general.get("generate") is True:
            generated.append(dataset_name)
            if "merge" in dataset_config:
                generated_merges.append(dataset_name)
    if generated and not generated_merges:
        print("Warning: config has generated datasets but no generated merge target for a final training corpus")
    if not generated:
        print("Warning: config has no dataset with general.generate: true")

def process_dataset(config_filename, test=False, postprocess_only=False):
    config = stuff.load_dictionary(config_filename)
    if "datasets" in config and postprocess_only is False:
        warn_if_no_final_merge_target(config)
        for dataset_name in config["datasets"]:
            generate_dataset(os.path.split(config_filename)[0],
                            os.path.split(config_filename)[1],
                            dataset_name, test=test)
    if "postprocess" in config:
        postprocess_datasets(config)

def process_dataset_single_task(config_yaml):
    config = stuff.load_dictionary(config_yaml)
    # Convenience helper for inspecting a single config file.
    path=os.path.split(config_yaml)[0]
    config_filename=os.path.split(config_yaml)[1]

    print(config_yaml)
    print(path)
    print(config_filename)
    print(os.path.join(path,config_filename))

    for dataset_name in config["datasets"]:
        dataset_config=get_expanded_config(path, config_filename, dataset_name)
        print(dataset_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default="dataset_config.yaml", help='Configuration to use (json/yaml)')
    parser.add_argument('--test', action='store_true', help='run in test mode (limit number of images)')
    parser.add_argument('--llm-test',  type=str, default=None, help='run in LLM test')
    parser.add_argument('--filter', action='store_true', help='filter datasets')
    parser.add_argument('--postprocess', action='store_true', help='postprocess datasets')
    parser.add_argument('--skip-gdrive-sync', action='store_true', help='skip model sync from gdrive at startup')
    opt = parser.parse_args()
    stuff.configure_root_logger(opt.logging)
    if opt.filter:
        config1=stuff.load_dictionary("/mldata/config/dataset/dataset_attributes_default.yaml")
        for s in ["coco", "openimages", "hoang", "weapons", "widerface"]:

            to_filter_small=config1["datasets"]["default"]["filter_classes"]["classes_to_filter_small"]
            config={"classes_to_filter_small":to_filter_small}
            build_task_filter_classes("/mldata/attr-v2-large/"+s+"-attr-v2-large/dataset.yaml", config)
        exit()

    if opt.llm_test is not None:
        llm_test(opt.llm_test)
        exit()

    if not opt.skip_gdrive_sync:
        model_folders=collect_model_folders_for_sync(opt.config)
        if len(model_folders)==0:
            print("Warning: no model folders found in config for gdrive sync")
        for model_folder in model_folders:
            print(f"Gdrive_sync: ensuring model folder {model_folder}")
            stuff.gdrive_sync_mldata(model_folder, "from_drive")

    process_dataset(opt.config, test=opt.test, postprocess_only=opt.postprocess)