import sys
import threading
import queue
import traceback

# Sentinel pushed downstream on end-of-stream so worker threads exit cleanly.
POISON = object()

# Base class for a pipeline stage running in its own thread. Messages arrive on
# a single bounded inputQueue (multiple upstreams may feed it); process() output
# is broadcast to every subscribed downstream queue. A node with N upstreams
# forwards POISON only after receiving N poisons, so joins drain correctly.
class Node:
    def __init__(self, name, queueSize=8):
        self.name = name
        self.inputQueue = queue.Queue(maxsize=queueSize)
        self.subscribers = []
        self.numUpstreams = 0
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)

    def subscribe(self, downstream):
        self.subscribers.append(downstream.inputQueue)
        downstream.numUpstreams += 1

    def emit(self, message):
        if message is None:
            return
        for q in self.subscribers:
            q.put(message)

    def process(self, message):
        # Override. Return a message, a list of messages, or None.
        raise NotImplementedError

    def _emitResult(self, result):
        if result is None:
            return
        if isinstance(result, list):
            for message in result:
                self.emit(message)
        else:
            self.emit(result)

    def _run(self):
        poisonsSeen = 0
        while True:
            message = self.inputQueue.get()
            if message is POISON:
                poisonsSeen += 1
                if poisonsSeen >= max(1, self.numUpstreams):
                    self._emitResult(self.onShutdown())
                    for q in self.subscribers:
                        q.put(POISON)
                    return
                continue
            # A failing frame must not kill the thread, or downstream nodes never
            # receive the poison pill and the graph deadlocks on shutdown.
            try:
                self._emitResult(self.process(message))
            except Exception:
                print(f"[{self.name}] error processing message; skipping frame", file=sys.stderr)
                traceback.print_exc()

    def onShutdown(self):
        # Override for nodes that flush buffered state at end-of-stream.
        return None

    def start(self):
        self.thread.start()

    def join(self):
        self.thread.join()


# A node with no upstreams that pumps messages from a generator, then poisons
# its subscribers. Subclasses implement produce() as a generator of messages.
class SourceNode(Node):
    def produce(self):
        raise NotImplementedError

    def _run(self):
        for message in self.produce():
            self.emit(message)
        for q in self.subscribers:
            q.put(POISON)
