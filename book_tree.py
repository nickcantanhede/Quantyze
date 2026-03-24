"""Quantyze book tree.

Module Description
==================
This module contains the binary search tree used to store order-book price
levels for Quantyze. Each node in the tree is a PriceLevel, and the tree
supports efficient insertion, lookup, deletion, and best-price access.

The tree is used by the order book to maintain the bid side and ask side of the
market:
- the maximum price in the bids tree is the best bid
- the minimum price in the asks tree is the best ask

This module is responsible only for the tree structure itself. It does not
handle order matching, transaction logging, or dataset parsing.

Copyright Information
===============================

Copyright (c) 2026 Cade McNelly, Nicolas Miranda Cantanhede,
Sahand Samadirand
"""

from price_level import PriceLevel
from dataclasses import dataclass


@dataclass()
class BookTree:
    """BST of PriceLevels for one side of the order book.

    Instance Attributes:
    - root: the root PriceLevel in this tree, or None if the tree is empty
    - side: which side of the order book this tree represents
    - size: the number of price levels currently stored in the tree

    Representation Invariants:
    - self.side in {'bid', 'ask'}
    - self.size >= 0
    """

    root: PriceLevel | None
    side: str
    size: int

    def __init__(self, side: str) -> None:
        """Initialize an empty book tree.

           Preconditions:
           - side in {'bid', 'ask'}

        """

        self.root = None
        self.side = side
        self.size = 0

    def is_empty(self) -> bool:
        """Return whether this tree contains no price levels."""

        return self.root is None

    def __len__(self) -> int:
        """Return the number of price levels stored in this tree."""

        return self.size

    def __contains__(self, price: float) -> bool:
        """Return whether this tree contains a PriceLevel at <price>."""

        curr = self.root

        while curr is not None:
            if price == curr.price:
                return True
            elif price < curr.price:
                curr = curr.left
            else:
                curr = curr.right

        return False

    def __getitem__(self, price: float) -> PriceLevel:
        """Return the PriceLevel stored at <price>.

        Raise KeyError if <price> is not stored in this tree.
        """

        curr = self.root

        while curr is not None:
            if price == curr.price:
                return curr
            elif price < curr.price:
                curr = curr.left
            else:
                curr = curr.right

        raise KeyError

    def insert(self, price_level: PriceLevel) -> None:
        """Insert <price_level> into this tree.

        If this tree already contains a node at that price, update or reuse the
        existing node instead of creating a duplicate price level.
        """

        if self.root is None:
            self.root = price_level
            self.size += 1
            return
        else:
            curr = self.root

            while curr is not None:
                if price_level.price == curr.price:
                    return
                elif price_level.price < curr.price:
                    if curr.left is None:
                        curr.left = price_level
                        self.size += 1
                        return

                    curr = curr.left
                else:
                    if curr.right is None:
                        curr.right = price_level
                        self.size += 1
                        return

                    curr = curr.right

    def delete(self, price: float) -> None:
        """Delete the PriceLevel at <price> from this tree.

        Raise KeyError if <price> is not stored in this tree.
        """

        if price in self:
            self.root = self._delete_node(self.root, price)
            self.size -= 1
        else:
            raise KeyError

    def _delete_node(self, node: PriceLevel | None, price: float) -> PriceLevel | None:
        """Return the updated subtree rooted at <node> after deleting <price>.

        If <price> is not stored in the subtree rooted at <node>, return <node>
        unchanged.

        This helper handles the three BST deletion cases:
        - deleting a leaf node
        - deleting a node with one child
        - deleting a node with two children, using the in-order successor

        Preconditions:
        - <node> is None or is the root of a valid BST of PriceLevels
        - all prices in the subtree rooted at <node> are unique
        """

        if node is None:
            return None
        elif price < node.price:
            node.left = self._delete_node(node.left, price)
            return node
        elif price > node.price:
            node.right = self._delete_node(node.right, price)
            return node
        else:

            if node.left is None and node.right is None:
                return None
            elif node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                sucessor = self._min_node(node.right)

                # Copy the sucessor data into node
                node.price = sucessor.price
                node.orders = sucessor.orders
                node.volume = sucessor.volume

                node.right = self._delete_node(node.right, sucessor.price)
                return node

    @staticmethod
    def _min_node(node: PriceLevel) -> PriceLevel:
        """Return the minimum-price node in the subtree rooted at <node>."""

        curr = node
        while curr.left is not None:
            curr = curr.left

        return curr

    @staticmethod
    def _max_node(node: PriceLevel) -> PriceLevel:
        """Return the maximum-price node in the subtree rooted at <node>."""

        curr = node
        while curr.right is not None:
            curr = curr.right

        return curr

    def best(self) -> PriceLevel | None:
        """Return the best PriceLevel in this tree.

        For a bid tree, this is the maximum-price node.
        For an ask tree, this is the minimum-price node.
        """

        if self.root is None:
            return None
        elif self.side == "bid":
            return self._max_node(self.root)
        else:
            return self._min_node(self.root)

    def inorder(self) -> list[PriceLevel]:
        """Return the price levels in this tree in ascending price order."""

        return self._inorder_node(self.root)

    def _inorder_node(self, node: PriceLevel | None) -> list[PriceLevel]:
        """Return the price levels in the subtree rooted at <node> in ascending price order.

        If <node> is None, return an empty list.
        """

        if node is None:
            return []
        else:
            return self._inorder_node(node.left) + [node] + self._inorder_node(node.right)

    def __repr__(self) -> str:
        """Return a string representation of this tree for debugging."""

        if self.root is None:
            root_price = None
        else:
            root_price = self.root.price

        return f"BookTree(side={self.side!r}, size={self.size}, root={root_price})"
