# Buffers messages per frameId until every required branch has arrived, then
# releases the assembled dict. Used by join nodes (localization, viz) to gather
# concurrent fan-out branches back together. drop() discards a frame whose
# upstream failed so it never lingers and late branches are ignored.
class FrameSynchronizer:
    def __init__(self, requiredKeys):
        self.requiredKeys = set(requiredKeys)
        self.buffer = {}
        self.dropped = set()

    def add(self, frameId, key, value):
        if frameId in self.dropped:
            return None
        entry = self.buffer.setdefault(frameId, {})
        entry[key] = value
        if self.requiredKeys <= set(entry.keys()):
            return self.buffer.pop(frameId)
        return None

    def drop(self, frameId):
        self.buffer.pop(frameId, None)
        self.dropped.add(frameId)


# Releases items in contiguous frameId order despite out-of-order arrival.
# Used by the video writer so frames are encoded sequentially. skip() marks a
# frame as never arriving so it cannot block the frames behind it.
class ReorderBuffer:
    def __init__(self, startId=0):
        self.nextId = startId
        self.pending = {}
        self.skipped = set()

    def push(self, frameId, item):
        if frameId < self.nextId or frameId in self.skipped:
            return []
        self.pending[frameId] = item
        return self._release()

    def skip(self, frameId):
        if frameId < self.nextId:
            return []
        self.pending.pop(frameId, None)
        self.skipped.add(frameId)
        return self._release()

    def _release(self):
        released = []
        while self.nextId in self.pending or self.nextId in self.skipped:
            if self.nextId in self.pending:
                released.append(self.pending.pop(self.nextId))
            self.skipped.discard(self.nextId)
            self.nextId += 1
        return released

    def flush(self):
        # Emit whatever remains in ascending order (end-of-stream drain).
        released = [self.pending[fid] for fid in sorted(self.pending)]
        self.pending.clear()
        return released
