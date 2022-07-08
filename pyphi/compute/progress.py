# progress.py
from asyncio import Event
from typing import Optional, Tuple

import ray
# For typing purposes
from ray.actor import ActorHandle
from tqdm.auto import tqdm



@ray.remote
class ProgressBarActor:
    """Keep track of progress on remote tasks."""

    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.finished = False
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental number of items that
        were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    def finish(self) -> None:
        """Sets the finished flag to True."""
        self.finished = True
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update` or `finish`, then returns a tuple of
        the number of updates since the last call to `wait_for_update`, and the
        total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter, self.finished

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    """Handles interactions with a remote ProgressBarActor."""

    progress_actor: ActorHandle
    total: Optional[int]
    desc: str
    pbar: tqdm

    def __init__(self, total: Optional[int], desc: str = ""):
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.desc = desc

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.desc, total=self.total)
        while True:
            delta, counter, finished = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if finished or counter >= self.total:
                pbar.close()
                return
