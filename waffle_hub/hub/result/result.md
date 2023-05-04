## TrainResult
::: waffle_hub.schema.result.TrainResult
    handler: python
    options:
        members:
            - best_ckpt_file
            - last_ckpt_file
            - metrics
        show_source: false

## EvaluateResult
::: waffle_hub.schema.result.EvaluateResult
    handler: python
    options:
        members:
            - metrics
        show_source: false

## InferenceResult
::: waffle_hub.schema.result.InferenceResult
    handler: python
    options:
        members:
            - predictions
            - draw_dir
        show_source: false

## ExportResult
::: waffle_hub.schema.result.ExportResult
    handler: python
    options:
        members:
            - export_file
        show_source: false