class Solution(object):
    def reverseList(self, head):
        if not head or not head.next:
            return head
        res = None
        while head:
            tmp = head.next
            head.next = res
            res = head
            head = tmp
        return res
