from src.stages import (
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

__all__ = [name for name in globals() if name.startswith("build_task_")]
