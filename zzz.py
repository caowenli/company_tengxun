# 算法题：
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def way(head):
    if not head or not head.next:
        return False
    s, f = head, head
    while f and f.next:
        f = f.next.next
        s = s.next
        if s == f:
            return True
    return False


def way1(head):
    res = None
    if not head or not head.next:
        return head
    cur = way1(head.next)
    head.next.next = head
    head.next = None
    return cur


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def way3(root):
    if not root or (not root.left and not root.right):
        return root
    root.left, root.right = root.right, root.left
    way3(root.left)
    way3(root.right)


def preorder(root):
    if not root:
        return []
    res = []

    def helper(root):
        if not root:
            return
        res.append(root.val)
        helper(root.left)
        helper(root.right)

    helper(root)
    return res


def preorder2(root):
    if not root:
        return []
    res = []
    queue = [root]
    while queue:
        node = queue.pop()
        res.append(node.val)
        if node.right:
            queue.append(node.right)
        if node.left:
            queue.append(node.left)
    return res


def way(lis):
    res = []

    def backtrack(lis, track):
        if len(track) == len(lis):
            res.append(track[:])
        for i in range(len(lis)):
            if lis[i] in track:
                continue
            track.append(lis[i])
            backtrack(lis, track)
            track.pop()
    backtrack(lis, [])
    return res
