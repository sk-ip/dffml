dffml dataflow create \
    -flow \
        '[{"seed": ["city"]}]'=operations:lookup_temperature.inputs.city \
        '[{"seed": ["month"]}]'=operations:lookup_temperature.inputs.month \
        '[{"seed": ["city"]}]'=operations:lookup_population.inputs.city \
        '[{"seed": ["state"]}]'=operations:lookup_population.inputs.state \
    -inputs \
        '["temperature", "population"]'=get_single_spec \
        -- \
        operations:lookup_population \
        operations:lookup_temperature \
        get_single | \
        tee preprocess_ops.json
