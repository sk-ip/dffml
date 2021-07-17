dffml dataflow createshort operations:lookup_population operations:lookup_temperature \
    -inputs \
        city \
        state \
        month \
    -op-lookup_temperature-city city \
    -op-lookup_temperature-month month \
    -op-lookup_population-city city \
    -op-lookup_population-state state | \
    tee short_ops.json
