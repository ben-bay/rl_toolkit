from gym.envs.registration import register

register(
        id='tictactoe-v0',
        entry_point='envs.custom_env_dir.tictactoe_env:TicTacToeEnv',
)

