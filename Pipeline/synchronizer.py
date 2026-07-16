# Buffers messages per frameId until every required branch has arrived, then
# releases the assembled dict. Used by join nodes (distance, localization, viz)
# to gather concurrent fan-out branches back together.
class FrameSynchronizer:
    def __init__(self, requiredKeys):
        self.requiredKeys = set(requiredKeys)
        self.buffer = {}

    def add(self, frameId, key, value):
        entry = self.buffer.setdefault(frameId, {})
        entry[key] = value
        if self.requiredKeys <= set(entry.keys()):
            return self.buffer.pop(frameId)
        return None


# Releases items in contiguous frameId order despite out-of-order arrival.
# Used by the video writer so frames are encoded sequentially.
class ReorderBuffer:
    def __init__(self, startId=0):
        self.nextId = startId
        self.pending = {}

    def push(self, frameId, item):
        self.pending[frameId] = item
        released = []
        while self.nextId in self.pending:
            released.append(self.pending.pop(self.nextId))
            self.nextId += 1
        return released

    def flush(self):
        # Emit whatever remains in ascending order (end-of-stream drain).
        released = [self.pending[fid] for fid in sorted(self.pending)]
        self.pending.clear()
        return released
