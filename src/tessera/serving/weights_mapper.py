"""Replay the runtime's explicit mapper API without depending on vLLM imports."""


def module_name_mapper(mapper):
    """Use the same name-only view the runtime hands quantization configs.

    New runtimes expose get_rename_mapper; earlier wrappers use
    get_unstacked_mapper, and plain tables expose neither. An exception from
    an existing method is a broken mapper and propagates rather than falling
    back to a different translation.
    """
    for name in ("get_rename_mapper", "get_unstacked_mapper"):
        unwrap = getattr(mapper, name, None)
        if unwrap is not None:
            return unwrap()
    return mapper
