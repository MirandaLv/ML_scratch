class ListNode:
    def __init__(self, key, val):
        self.val = val
        self.key = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = dict()
        # Don't forget to create a new dummy node for both head and tail
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        # Also point the head and tail dummy nodes to each other
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.dic: return -1

        node = self.dic[key]  # get the node using key to search
        self.remove(node)  # remove the node from current position
        self.add(node)  # add the node to the end of the linked list since it is most recently fetched
        return node.val

    def put(self, key: int, value: int) -> None:
        # if the key exist, remove the node first, and then add a new node later
        if key in self.dic:
            old_node = self.dic[key]
            self.remove(old_node)

        node = ListNode(key, value)
        self.dic[key] = node
        self.add(node)

        # if the key does not exist, and check whether it exist capacity
        if len(self.dic) > self.capacity:
            node_to_delete = self.head.next
            self.remove(node_to_delete)
            del self.dic[node_to_delete.key]  # don't forget the delete the delted node's key

    def add(self, node):
        # add a node to the end of the linked list whenever adding a new key or
        # updating an existing one
        previous_end = self.tail.prev  # previous end node
        previous_end.next = node  # the new node is added to the end first
        node.prev = previous_end
        node.next = self.tail
        self.tail.prev = node

    def remove(self, node):
        # remove a node from the linked list
        # perform removal when we update/fetch an exisitng key, or when the data structure
        # exceeding the capacity
        node.prev.next = node.next
        node.next.prev = node.prev

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)