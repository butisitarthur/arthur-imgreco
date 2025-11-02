"""
Hierarchical ID utilities for Arthur Image Recognition 2.0

This module provides consistent hierarchical ID generation and validation
across the application, implementing the artist_id.entry_id.view_id pattern.
"""

import uuid
from typing import Optional, Tuple

from core.logging import get_logger

logger = get_logger(__name__)

# Standard namespace UUID for hierarchical ID generation
HIERARCHICAL_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')


def generate_vector_id(artist_id: str, entry_id: str, view_id: str) -> str:
    """
    Generate a deterministic UUID5 from hierarchical components.
    
    Args:
        artist_id: Unique artist identifier
        entry_id: Unique artwork/entry identifier  
        view_id: View identifier (main, detail, side, etc.)
    
    Returns:
        UUID string that's deterministic for the same input
        
    Example:
        >>> generate_vector_id("vangogh", "starry_night", "main")
        "e5e60e43-943a-52ed-b281-6dbcc4ca5268"
    """
    hierarchical_id = f"{artist_id}.{entry_id}.{view_id}"
    return str(uuid.uuid5(HIERARCHICAL_NAMESPACE, hierarchical_id))


def create_hierarchical_id(artist_id: str, entry_id: str, view_id: str) -> str:
    """
    Create a hierarchical ID string from components.
    
    Args:
        artist_id: Unique artist identifier
        entry_id: Unique artwork/entry identifier  
        view_id: View identifier
        
    Returns:
        Hierarchical ID in format: artist_id.entry_id.view_id
    """
    return f"{artist_id}.{entry_id}.{view_id}"


def parse_hierarchical_id(hierarchical_id: str) -> Tuple[str, str, str]:
    """
    Parse a hierarchical ID string into components.
    
    Args:
        hierarchical_id: String in format artist_id.entry_id.view_id
        
    Returns:
        Tuple of (artist_id, entry_id, view_id)
        
    Raises:
        ValueError: If the ID format is invalid
    """
    parts = hierarchical_id.split('.')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid hierarchical ID format. Expected 'artist_id.entry_id.view_id', "
            f"got '{hierarchical_id}'"
        )
    
    return parts[0], parts[1], parts[2]


def validate_hierarchical_components(
    artist_id: Optional[str],
    entry_id: Optional[str], 
    view_id: Optional[str]
) -> bool:
    """
    Validate that hierarchical components are complete and valid.
    
    Args:
        artist_id: Artist identifier
        entry_id: Entry identifier
        view_id: View identifier
        
    Returns:
        True if all components are valid
    """
    return all([
        artist_id and isinstance(artist_id, str) and artist_id.strip(),
        entry_id and isinstance(entry_id, str) and entry_id.strip(),  
        view_id and isinstance(view_id, str) and view_id.strip()
    ])


def resolve_vector_id(
    vector_id: Optional[str] = None,
    artist_id: Optional[str] = None,
    entry_id: Optional[str] = None,
    view_id: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Resolve the final vector ID from either direct ID or hierarchical components.
    
    Args:
        vector_id: Direct vector ID (takes precedence if provided)
        artist_id: Artist identifier for hierarchical generation
        entry_id: Entry identifier for hierarchical generation
        view_id: View identifier for hierarchical generation
        
    Returns:
        Tuple of (final_vector_id, hierarchical_id_or_none)
        
    Raises:
        ValueError: If neither vector_id nor complete hierarchical components provided
    """
    # If vector_id is explicitly provided, use it
    if vector_id:
        logger.debug("Using provided vector_id", vector_id=vector_id)
        return vector_id, None
        
    # Try to generate from hierarchical components
    if validate_hierarchical_components(artist_id, entry_id, view_id):
        hierarchical_id = create_hierarchical_id(artist_id, entry_id, view_id)
        generated_vector_id = generate_vector_id(artist_id, entry_id, view_id)
        
        logger.debug(
            "Generated vector_id from hierarchical components",
            hierarchical_id=hierarchical_id,
            vector_id=generated_vector_id
        )
        
        return generated_vector_id, hierarchical_id
    
    # Neither option worked
    raise ValueError(
        "Either 'vector_id' must be provided, or all three of 'artist_id', 'entry_id', "
        "and 'view_id' must be provided for auto-generation. "
        f"Found: vector_id={vector_id}, artist_id={artist_id}, entry_id={entry_id}, view_id={view_id}"
    )