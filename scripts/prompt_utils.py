#!/usr/bin/env python3
"""
Utility functions for loading and formatting prompt templates.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional


def load_prompt_templates(template_file: str = "./prompt_templates.yaml") -> Dict:
    """Load prompt templates from YAML file."""
    template_path = Path(template_file)
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {template_file}")
    
    with open(template_path, "r") as f:
        templates = yaml.safe_load(f)
    
    return templates


def get_prompt_style(templates: Dict, style: str = "default") -> Dict:
    """
    Get a specific prompt style from templates.
    
    Args:
        templates: Loaded template dictionary
        style: Style name ("default", "concise", "detailed", etc.)
    
    Returns:
        Dictionary with prompt templates for the specified style
    """
    if style == "default":
        return {
            "system_message": templates.get("system_message", ""),
            "instruction_template": templates.get("instruction_template", ""),
            "user_message_template": templates.get("user_message_template", ""),
            "output_group_template": templates.get("output_group_template", ""),
        }
    
    # Get alternative style
    alternative_styles = templates.get("alternative_styles", {})
    if style not in alternative_styles:
        raise ValueError(
            f"Style '{style}' not found. Available styles: default, {', '.join(alternative_styles.keys())}"
        )
    
    return alternative_styles[style]


def format_instruction(template: str, words: List[str], remaining_count: Optional[int] = None) -> str:
    """Format instruction template with words."""
    words_str = ", ".join(words)
    if remaining_count is not None:
        return template.format(remaining_count=remaining_count, words=words_str).strip()
    return template.format(words=words_str).strip()


def format_user_message(template: str, words: List[str], remaining_count: Optional[int] = None) -> str:
    """Format user message template with words."""
    words_str = ", ".join(words)
    if remaining_count is not None:
        return template.format(remaining_count=remaining_count, words=words_str).strip()
    return template.format(words=words_str).strip()


def format_output_group(template: str, group_words: List[str], category: str, difficulty: str) -> str:
    """Format output group template with group information."""
    words_str = ", ".join(group_words)
    return template.format(
        group_words=words_str,
        category=category,
        difficulty=difficulty
    )


def create_chat_messages(
    words: List[str],
    system_message: str,
    user_message_template: str,
    include_assistant_response: Optional[str] = None,
    remaining_count: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Create chat messages for the model.
    
    Args:
        words: List of words (16 for all-at-once, variable for iterative)
        system_message: System message content
        user_message_template: Template for user message
        include_assistant_response: Optional assistant response to include
        remaining_count: Optional count of remaining words (for iterative approach)
    
    Returns:
        List of message dictionaries
    """
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": format_user_message(user_message_template, words, remaining_count)
        }
    ]
    
    if include_assistant_response:
        messages.append({
            "role": "assistant",
            "content": include_assistant_response
        })
    
    return messages

