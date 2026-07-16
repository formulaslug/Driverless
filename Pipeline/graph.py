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

    def join(self):
        for node in self.nodes:
            node.join()

    def run(self):
        self.start()
        self.join()
