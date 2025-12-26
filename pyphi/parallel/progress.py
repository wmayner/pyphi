# parallel/progress.py
"""Progress bars for distributed computations."""

from asyncio import Event
from time import time
from typing import Optional, Tuple, TYPE_CHECKING

from ..deferred.ray import ray

if TYPE_CHECKING:
    import ray
    from ray import ActorHandle

from tqdm.auto import tqdm

from ..conf import fallback


@ray.remote
class ProgressBarActor:
    """Keep track of progress on remote tasks."""

    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.finished = False
        self.interrupted = False
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

    def finish(self, interrupted=False) -> None:
        """Sets the finished flag to True."""
        self.finished = True
        self.interrupted = interrupted
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
        return saved_delta, self.counter, self.finished, self.interrupted


@ray.remote
def wait_then_finish(progress_bar, object_refs):
    ray.wait(object_refs, num_returns=len(object_refs))
    progress_bar.actor.finish.remote()


class ProgressBar:
    """Handles interactions with a remote ProgressBarActor."""

    _actor: "ActorHandle"
    total: Optional[int]
    desc: str
    pbar: tqdm

    def __init__(self, total: Optional[int], desc: str = ""):
        self._actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.desc = desc

    @property
    def actor(self) -> "ActorHandle":
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self._actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.desc, total=self.total)
        total = fallback(self.total, float("inf"))
        while True:
            delta, counter, finished, interrupted = ray.get(
                self.actor.wait_for_update.remote()
            )
            pbar.update(delta)
            if finished or counter >= total:
                # Explicitly set total since finish signal may arrive before the
                # counter is updated
                if not interrupted:
                    pbar.n = total
                    pbar.refresh()
                pbar.close()
                return


# Minimum time between progress bar updates (seconds)
THROTTLE_TIME = 0.01


def throttled_update(progress_bar, items):
    """Throttle progress update calls so the scheduler isn't overwhelmed."""
    num_since_last_update = 0
    last_update = time()
    for item in items:
        current_time = time()
        num_since_last_update += 1
        if current_time - last_update > THROTTLE_TIME:
            last_update = current_time
            progress_bar.actor.update.remote(num_since_last_update)
            num_since_last_update = 0
        yield item
    if num_since_last_update > 0:
        progress_bar.actor.update.remote(num_since_last_update)
