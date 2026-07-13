"""Network Situation Report drafter.

Rebuilds the weekly NSR that EUROCONTROL publishes at /api/situation_reports:
numbers come from the Data App API, causes come from the archived NOP tactical
updates, and the LLM only supplies connective prose around facts it may not alter.
"""
