import sys
import threading
import queue
import traceback

from messages import FailedFrameMessage

# Sentinel pushed downstream on end-of-stream so worker threads exit cleanly.
POISON = object()

# Base class for a pipeline stage running in its own thread. Messages arrive on
# a single bounded inputQueue (multiple upstreams may feed it); process() output
# is broadcast to every subscribed downstream queue. A node with N upstreams
# forwards POISON only after receiving N poisons, so joins drain correctly.
# A frame whose process() raises is announced downstream as a FailedFrameMessage
# so join nodes can discard it via skipFrame().
class Node:
    def __init__(self, name, queueSize=8):
        self.name = name
        self.inputQueue = queue.Queue(maxsize=queueSize)
        self.subscribers = []
        self.numUpstreams = 0
        self.stopEvent = threading.Event()
        self.failedFrames = set()
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

    def skipFrame(self, frameId):
        # Override in nodes that buffer per-frame state; may return outputs released by the skip.
        return None

    def _emitResult(self, result):
        if result is None:
            return
        if isinstance(result, list):
            for message in result:
                self.emit(message)
        else:
            self.emit(result)

    def _shutdown(self):
        try:
            self._emitResult(self.onShutdown())
        except Exception:
            print(f"[{self.name}] error during shutdown", file=sys.stderr)
            traceback.print_exc()
        finally:
            for q in self.subscribers:
                q.put(POISON)

    def _run(self):
        poisonsSeen = 0
        while True:
            message = self.inputQueue.get()
            if message is POISON:
                poisonsSeen += 1
                if poisonsSeen >= max(1, self.numUpstreams):
                    self._shutdown()
                    return
                continue
            try:
                if isinstance(message, FailedFrameMessage):
                    self._handleFailedFrame(message)
                else:
                    self._emitResult(self.process(message))
            except Exception:
                print(f"[{self.name}] error processing frame {getattr(message, 'frameId', '?')}; skipping", file=sys.stderr)
                traceback.print_exc()
                frameId = getattr(message, 'frameId', None)
                if frameId is not None:
                    self._handleFailedFrame(FailedFrameMessage(frameId, self.name))

    def _handleFailedFrame(self, failure):
        if failure.frameId in self.failedFrames:
            return
        self.failedFrames.add(failure.frameId)
        try:
            self._emitResult(self.skipFrame(failure.frameId))
        except Exception:
            print(f"[{self.name}] error discarding frame {failure.frameId}", file=sys.stderr)
            traceback.print_exc()
        self.emit(failure)

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
        try:
            for message in self.produce():
                if self.stopEvent.is_set():
                    break
                self.emit(message)
        except Exception:
            print(f"[{self.name}] source failed", file=sys.stderr)
            traceback.print_exc()
        finally:
            for q in self.subscribers:
                q.put(POISON)
