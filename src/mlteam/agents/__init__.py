"""Agents: each one is a Tau ``AgentHarness`` with its own prompt and tools.

The harness code in ``mlteam.team`` decides who runs and in what order. Agents
never call each other; they hand over artefacts (a contract, a review).
"""
