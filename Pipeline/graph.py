# Owns the set of nodes and starts/joins their threads. Wiring (subscribe) is
# done by the caller before start(); the graph just manages lifecycle.
class Graph:
    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)
        return node

    def start(self):
        for node in self.nodes:
            node.start()

    def stop(self):
        for node in self.nodes:
            node.stopEvent.set()

    def join(self):
        for node in self.nodes:
            while node.thread.is_alive():
                node.thread.join(timeout=0.5)

    # Ctrl-C stops the sources and lets the remaining frames drain.
    def run(self):
        self.start()
        try:
            self.join()
        except KeyboardInterrupt:
            print("Interrupted; draining pipeline...")
            self.stop()
            while True:
                try:
                    self.join()
                    break
                except KeyboardInterrupt:
                    print("Still draining; waiting for the video to finalize...")
