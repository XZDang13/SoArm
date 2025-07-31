import gymnasium

gymnasium.register(
    id="REACH-v0",
    entry_point=f"{__name__}.reach_env:ReachTask"
)