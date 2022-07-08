import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json

class Node:
    def __init__(self, iD = None, next_node = None):
        self.iD = iD
        self.next = next_node

class LinkedList:
    def __init__(self,head = None):
        self.head = head
        self.circular_lenght = None

    def is_head_undefined(self):
        return self.head is None

    def __str__(self):
        if self.is_head_undefined():
            return "Empty linked list"
        actual_node = self.head
        list_str = ''
        if self.circular_lenght is None:
            while actual_node:
                list_str += str(actual_node.iD)+'-> '
                actual_node = actual_node.next
        else:
            for i in range((self.circular_lenght+1)*2):
                list_str += str(actual_node.iD)+'-> '
                actual_node = actual_node.next
        return list_str

    def find_iD(self, iD):
        actual_node = self.head
        while actual_node:
            if actual_node.iD == iD:
                print('Found!')
                return actual_node
            actual_node = actual_node.next

    def add_end(self,iD):
        if self.is_head_undefined():
            self.head = Node(iD = iD)
        else:    
            actual_node = self.head
            while actual_node:
                if actual_node.next is None:
                    actual_node.next = Node(iD = iD)
                    break
                actual_node = actual_node.next

    def make_circular(self):
        actual_node = self.head
        i = 0
        while actual_node:
            if actual_node.next is None:
                actual_node.next = self.head
                self.circular_lenght = i
                print(self.circular_lenght)
                break
            actual_node = actual_node.next
            i+=1

    def get_next_n_iDs(self, iD, n):
        actual_node = self.find_iD(iD)
        output = dict()
        for i in range(n):
            output[actual_node.iD] = (i,i+n)
            actual_node = actual_node.next
        return output
    
    def get_next_n_nodes(self, iD, n):
        actual_node = self.find_iD(iD)
        output = []
        for i in range(n):
            output.append(actual_node)
            actual_node = actual_node.next
        return output