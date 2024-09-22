import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class State:
    last_saved_example_num: int
    current_shard_id: int
    total_processed_examples: int

    def save_state(self, state_path: str):
        """Save the state to a file.
        Args:
            state_path (str): The path to save the state to.

        """
        with open(state_path, "w") as f:
            json.dump(
                {
                    "last_saved_example_num": self.last_saved_example_num,
                    "current_shard_id": self.current_shard_id,
                },
                f,
            )
        logger.info(f"State saved to {state_path}")
