import gymnasium

gymnasium.register(
    id="REACH-v0",
    entry_point=f"{__name__}.reach_env:ReachTask"
)

gymnasium.register(
    id="STACK-v0",
    entry_point=f"{__name__}.stack_env:StackTask"
)

gymnasium.register(
    id="STACK-Vis-v0",
    entry_point=f"{__name__}.stack_visual_env:StackTask"
)