import os
import random
from tqdm import tqdm
import stuff
import src.dataset_util as dsu

def build_task_import(processor, import_config, test=False):
    """
    import
    - use one of the import functions to load the initial images and GTs from a dataset
    - the classes/annotation are remapped to our target set
    - the files are renamed
    """
    task=processor.task
    class_names=processor.class_names
    name=processor.dataset_name

    processor.log(f"import; class_names={class_names}")

    loader=import_config["loader"]

    max_width=dsu.get_param(processor.task, import_config, "max_width", 1920)
    max_height=dsu.get_param(processor.task, import_config, "max_height", 1920)
    max_bpp=dsu.get_param(processor.task, import_config, "max_bpp", 3.0)
    yuv420_frac=dsu.get_param(processor.task, import_config, "yuv420", 0.0)
    fix_orientation=dsu.get_param(processor.task, import_config, "fix_orientation", False)
    yuv420=random.random()<yuv420_frac

    max_images=dsu.get_param(processor.task, import_config, "max_images", 10000000)
    add_exif=dsu.get_param(processor.task, import_config, "add_exif", True)
    filter_single_large=dsu.get_param(processor.task, import_config, "filter_single_large", False)
    enable_ignore_regions=dsu.get_param(processor.task, import_config, "enable_ignore_regions", True)

    if enable_ignore_regions:
        class_name_ignore=class_names+["ignore"]
        ignore_class=class_name_ignore.index("ignore")
    else:
        class_name_ignore=-1
        ignore_class=None

    loader_fn=dsu.get_loader(loader)
    o=loader_fn(class_names=class_name_ignore, task=task, ds_params=import_config)

    processor.log("origin info: "+o.get_info())
    maps=o.get_category_maps()
    for c in class_name_ignore:
        if c in maps:
            o.add_category_mapping(maps[c], c)

    ids=o.get_image_ids()
    if test and loader=="PersonPath22Loader":
        # In test mode for PersonPath22, sample frames from the whole split instead
        # of always taking the first contiguous chunk.
        test_shuffle_seed=int(dsu.get_param(processor.task, import_config, "test_shuffle_seed", 42))
        ids=list(ids)
        random.Random(test_shuffle_seed).shuffle(ids)
        processor.log(
            f"test mode: shuffled PersonPath22 ids with seed={test_shuffle_seed}"
        )
    if len(ids) != len(set(ids)):
        print("WARNING: Duplicate ids")
    processor.log(f"found {len(ids)} ids, limiting to {max_images}")

    base=name+"_"+task[0]

    index=0
    no_annotations=0
    filtered_single_large=0

    max_index=min(len(ids), max_images)
    if test:
        max_index=min(max_index, 250)

    desc=name+"/"+task+"/Generate"
    add_failed=0
    with tqdm(total=max_index, desc=desc.ljust(45), colour="#008000", smoothing=0.01) as pbar:
        for k,i in enumerate(ids):
            pbar.update(max(index, (k*max_index)//len(ids))-pbar.n)
            img_path=o.get_img_path(i)
            if img_path is None or not os.path.isfile(img_path):
                continue
            dets=o.get_annotations(i)
            if dets is None:
                no_annotations+=1
                continue

            if filter_single_large:
                if len(dets)==1:
                    if stuff.box_a(dets[0]["box"])>0.20:
                        filtered_single_large+=1
                        continue

            n=base+f"{index:07d}"
            if False==processor.add(n, img_path, dets,
                                    add_exif=add_exif or test,
                                    ignore_class=ignore_class,
                                    max_width=max_width,
                                    max_height=max_height,
                                    max_bpp=max_bpp,
                                    yuv420=yuv420,
                                    fix_orientation=fix_orientation):
                add_failed+=1
            index+=1
            pbar.update(1)

            if index>=max_index:
                break
        pbar.update(max_index-pbar.n)

    processor.log(f"imported {index} images {add_failed} failed")
    if no_annotations!=0:
        processor.log(f"import_dataset: filtered {no_annotations} images due to 'None' annotations")
    if filtered_single_large!=0:
        processor.log(f"import_dataset: filtered {filtered_single_large} images due to 'filtered_single_large'")
    processor.reload_files()
    processor.log_stats()

    dedup=dsu.get_param(processor.task, import_config, "dedup", False)
    if dedup:
        build_task_dedup(processor)

def build_task_dedup(processor):
    desc=processor.config_name+"/"+processor.task+"/dedup"
    num=processor.num_files
    deduper=stuff.ImgDedup()
    num_deleted=0
    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a010", smoothing=0.01):
        img=processor.get_img_path(i)
        r=deduper.test(img)
        if r.duplicate==True:
            processor.delete(i)
            num_deleted+=1
    processor.log(f"Deup deleted {num_deleted} images from {num}, {100.0*num_deleted/(num+1e-7):3.2f}%")
    processor.reload_files()
    processor.log_stats()

def build_task_sample(processor, sample_config, test=False):
    """
    sample
    - Keep only a subset of images in the dataset (useful for quick experiments).
    - This stage is deterministic when `seed` is set and `method: random` is used.

    Config options:
      max_images: int (required)            # number of images to keep
      method: "random"|"first"|"last"       # selection strategy (default: random)
      seed: int                             # RNG seed for method=random (default: 0)
      rename: bool                          # if true, rename kept images sequentially (default: true)
      base_name: str                        # output name prefix for renaming (default: <config>_<t>s)
    """
    if sample_config is None:
        return

    max_images=dsu.get_param(processor.task, sample_config, "max_images", None)
    if max_images is None:
        return

    if test:
        max_images=min(int(max_images), 50)

    num=processor.num_files
    if num<=max_images:
        processor.log(f"sample: nothing to do ({num} <= {max_images})")
        return

    method=dsu.get_param(processor.task, sample_config, "method", "random")
    seed=int(dsu.get_param(processor.task, sample_config, "seed", 0))
    do_rename=bool(dsu.get_param(processor.task, sample_config, "rename", True))

    indices=list(range(num))
    if method=="random":
        rng=random.Random(seed)
        rng.shuffle(indices)
        keep=set(indices[:max_images])
    elif method=="first":
        keep=set(indices[:max_images])
    elif method=="last":
        keep=set(indices[-max_images:])
    else:
        raise ValueError(f"sample: unknown method '{method}' (expected random|first|last)")

    desc=processor.config_name+"/"+processor.task+f"/sample({method})"
    for i in tqdm(range(num), desc=desc.ljust(45), colour="#10a0ff", smoothing=0.01):
        if i not in keep:
            processor.delete(i)

    processor.reload_files()

    if do_rename:
        base_name=dsu.get_param(processor.task, sample_config, "base_name", None)
        if base_name is None:
            base_name=processor.config_name+"_"+processor.task[0]+"s"
        desc=processor.config_name+"/"+processor.task+"/sample renaming"
        for i in tqdm(range(processor.num_files), desc=desc.ljust(45), colour="#20b0ff", smoothing=0.01):
            processor.rename(i, base_name+f"{i:07d}")
        processor.reload_files()

    processor.log(f"sample: reduced {num} -> {processor.num_files} (method={method}, seed={seed}, rename={do_rename})")
    processor.log_stats()
