"""
Pokemon AI Agents

Provides wrappers to run trained models in various environments.
"""

try:
    from pokemon_ai.agents.poke_env_agent import PokemonAIAgent, create_agent
    __all__ = ["PokemonAIAgent", "create_agent"]
except ImportError:
    # poke-env not installed
    __all__ = []
