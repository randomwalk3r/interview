
Questions to ask before starting solving problem:
1. What is time complexity? (Based on it you can get an idea what method to use)
2. What is space complexity?
3. Write what would be the solution? Try to work out based on example

***
Add test cases for your solution!!!


```python
import unittest

class TestCases(unittest.TestCase):
    def test_upper(self):
        self.assertEqual(5, 5)
        
#if __name__ == '__main__':
#    unittest.main()
```

<h1>Anagram</h1> is a word or phrase formed by rearranging the letters of a different word or phrase
<p>
s1 ="listen"
s2 ="silent"


```python
"""
Method 1: use sorted() and compare sorted methods
"""
def isAnagram1(w1, w2):
    return True if sorted(w1) == sorted(w2) else False

"""
Method 2: Count characters using preallocated list
"""
NUM_OF_CHARS = 256
def isAnagram2(w1, w2):
    count = [0] * 256
    for s1, s2 in zip(w1, w2):
        count[ord(s1)] += 1
        count[ord(s2)] -= 1
            
    for v in count:
        if v != 0:
            return False
    return True
s1 ="listen"
s2 ="silent"

if isAnagram1(s1, s2):
    print("The strings are anagrams.") 
else:
    print("The strings aren't anagrams.") 
    
if isAnagram2(s1, s2):
    print("The strings are anagrams.") 
else:
    print("The strings aren't anagrams.")

```

    The strings are anagrams.
    The strings are anagrams.
    

<h1>Palindrome</h1> reads the same backwards as forwards, e.g. madam


```python
'''
Check if a string is palindrome or not
'''
def isPolindrome(text):
    mid = len(text)/2
    for i in range(mid):
        if  text[i] != text[len(text)-1-i]:
            return False
    return True

s = "malayalam"
if isPolindrome(s):
    print("Yes")
else:
    print("No")
```

    Yes
    

<h1> Other </h1>


```python
"""
Find the longest substring with k unique characters in a given string
"""


def findLongest(s, k):
    result = ""
    for i in range(len(s)):
        end = i
        unique = []
        unique.append(s[i])
        for n in range(i+1, len(s)):
            if s[n] in unique:
                end += 1
                continue
            else:
                if len(unique) == k:
                    break
                unique.append(s[n])
                end += 1
        if len(result) < len(s[i:end+1]):
            result = s[i:end+1]
    return result

s = "aabbcc"
k = 3

assert findLongest(s, k) == s
```


```python
"""
Look through array to find pairs that sum to k
solution doesn't work but in right direction

"""

def findPair(A, k):
    pairCount = 0
    neededPairs = {} #this cannot be lagers than sum
    for n in A:
        if n in neededPairs.keys():
            pairCount += neededPairs[n] 
            neededPairs[k-n] +=  1 
        else:
           neededPairs[k-n] =  1 
        
    print pairCount
    return pairCount
    
A = [1, 5, 7, -1, 5]
assert findPair(A, 6) == 3

A = [1, 1, 1, 1]
assert findPair(A, 2) == 6
```


```python
"""
Find the second largest number in the array.
"""
# Python program to
# find second largest 
# element in an array
 
 
# Function to print the
# second largest elements 
def print2largest(arr,arr_size):
 
    # There should be atleast
        # two elements 
    if (arr_size < 2):
     
        print(" Invalid Input ")
        return
     
 
    first = second = -2147483648
    for i in range(arr_size):
     
        # If current element is
                # smaller than first
        # then update both
                # first and second 
        if (arr[i] > first):
         
            second = first
            first = arr[i]
         
 
        # If arr[i] is in
                # between first and 
        # second then update second 
        elif (arr[i] > second and arr[i] != first):
            second = arr[i]
     
    if (second == -2147483648):
        print("There is no second largest element")
    else:
        print("The second largest element is", second)
 
 
# Driver program to test
# above function 
arr = [12, 35, 1, 10, 34, 1]
n =len(arr)
 
print2largest(arr, n)
```

    ('The second largest element is', 34)
    


```python
"""
First question : Given a sorted array which has been rotated k times , need to find k?
"""
# Python3 program to find number
# of rotations in a sorted and
# rotated array.
 
# Returns count of rotations for
# an array which is first sorted
# in ascending order, then rotated
def countRotations(arr, n):
 
    # We basically find index
    # of minimum element
    min = arr[0]
    for i in range(0, n):
     
        if (min > arr[i]):
         
            min = arr[i]
            min_index = i
         
    return min_index;
 
 
# Driver code
arr = [15, 18, 2, 3, 6, 12]
n = len(arr)
print(countRotations(arr, n))
```

    2
    

***
Find max sum


```python
"""
Best Time to Buy and Sell Stock
If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
"""

def maxProfit(prices):
    maxToHere = maxSoFar = 0
    pricesDiff = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    for p in pricesDiff:
        maxToHere = max(0, maxToHere + p)
        maxSoFar = max(maxSoFar, maxToHere)
    
    print maxSoFar
    return maxSoFar

prices =  [7, 1, 5, 3, 6, 4]
assert maxProfit(prices) == 5
```

    5
    

<h3>Maximum path sum between two leaves of a binary tree</h3>


```python
#Given a binary tree find the maximum sum from leaf to leaf
# Python program to find maximumpath sum between two leaves
# of a binary tree
 
INT_MIN = -2**32
 
# A binary tree node
class Node:
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def maxPathSumUtil(root, res):
    if root is None:
        return 0
     
    if root.left is None and root.right is None:
        return root.data
     
    # Find maximumsum in left and righ subtree. Also
    # find maximum root to leaf sums in left and righ 
    # subtrees ans store them in ls and rs
    ls = maxPathSumUtil(root.left, res)
    rs = maxPathSumUtil(root.right, res)
 
    # If both left and right children exist
    if root.left is not None and root.right is not None:
 
        # update result if needed
        res[0] = max(res[0], ls + rs + root.data)
 
        # Return maximum possible value for root being 
        # on one side
        return max(ls, rs) + root.data
         
    # If any of the two children is empty, return
    # root sum for root being on one side
    if root.left is None:
        return rs + root.data
    else:
        return ls + root.data
 
# The main function which returns sum of the maximum 
# sum path betwee ntwo leaves. THis function mainly 
# uses maxPathSumUtil()
def maxPathSum(root):
        res = [INT_MIN]
        maxPathSumUtil(root, res)
        return res[0]
 
 
# Driver program to test above function
root = Node(-15)
root.left = Node(5)
root.right = Node(6)
root.left.left = Node(-8)
root.left.right = Node(1)
root.left.left.left = Node(2)
root.left.left.right = Node(6)
root.right.left = Node(3)
root.right.right = Node(9)
root.right.right.right= Node(0)
root.right.right.right.left = Node(4)
root.right.right.right.right = Node(-1)
root.right.right.right.right.left = Node(10)
 
print "Max pathSum of the given binary tree is", maxPathSum(root)
```

    Max pathSum of the given binary tree is 27
    

<h3> Count stairs </h3>


```python
# -*- coding: utf-8 -*-
"""
A kid wants to travel up n steps. She can only go 1, 2 or 3 steps at a time. How many different ways can she get to the nth

"""
'''
Method 1: Using recursion. 
        Time complexity O(n^3)
'''
def stepCount(n):
    if n < 0:
        return 0
    elif n == 0:
        return 1
    else:
        return stepCount(n-1) + stepCount(n-2) + stepCount(n-3)

'''
Method 2: Using memorization table
    Time complexity O(N*M)
'''
def stepCountDP(n):
    memo = [-1] * (n+1)
    return stepCountDPUtil(n, memo)
    
def stepCountDPUtil(n, memo):
    if n < 0:
        return 0
    elif n == 0:
        return 1
    elif memo[n] > -1:
        return memo[n]
    else:
        memo[n] = stepCountDPUtil(n-1, memo) + stepCountDPUtil(n-2, memo) + stepCountDPUtil(n-3, memo)
        return memo[n]
    
assert stepCount(4) == 7
assert stepCountDP(4) == 7
```

<h3>Longest Palindromic Subsequence</h3>


```python
#A Dynamic Programming based Python program for LPS problem
# Returns the length of the longest palindromic subsequence in seq
def lps(str):
    n = len(str)
 
    # Create a table to store results of subproblems
    L = [[0 for x in range(n)] for x in range(n)]
 
    # Strings of length 1 are palindrome of length 1
    for i in range(n):
        L[i][i] = 1
 
    # Build the table. Note that the lower diagonal values of table are
    # useless and not filled in the process. The values are filled in a
    # manner similar to Matrix Chain Multiplication DP solution (See
    # https://www.geeksforgeeks.org/dynamic-programming-set-8-matrix-chain-multiplication/
    # cl is length of substring
    for cl in range(2, n+1):
        for i in range(n-cl+1):
            j = i+cl-1
            if str[i] == str[j] and cl == 2:
                L[i][j] = 2
            elif str[i] == str[j]:
                L[i][j] = L[i+1][j-1] + 2
            else:
                L[i][j] = max(L[i][j-1], L[i+1][j]);
 
    return L[0][n-1]
 
# Driver program to test above functions
seq = "GEEKS FOR GEEKS"
n = len(seq)
print("The length of the LPS is " + str(lps(seq)))
 
# This code is contributed by Bhavya Jain
```

    The length of the LPS is 7
    

<h2/>Pangrams


```python
"""
"The quick brown fox jumps over the lazy dog" repeatedly. 
 This sentence is known as a pangram because it contains every letter of the alphabet.
"""
NUM_OF_CHARS = 256
start = ord("a")
end = ord("z")
def pangrams(s):
    alphabet = [0] * NUM_OF_CHARS
    for sym in s.lower():
        alphabet[ord(sym)] = 1
    return "pangram" if sum(alphabet[start:end+1]) == 26 else "not pangram"


s = "We promptly judged antique ivory buckles for the next prize"
assert pangrams(s) == "pangram"

s = "We promptly judged antique ivory buckles for the prize"
assert pangrams(s) == "not pangram"
```

<h2/>Common Child


```python
"""
Questions to ask:

"""
def commonChild(s1, s2):
    n = len(s1)
    m = len(s2)
    L = [[0] * (n+1) for i in range(m+1)]
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                L[i][j] = 1 + L[i-1][j-1]
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])


    # Following code is used to print LCS
    index = L[n][m]
 
    # Create a character array to store the lcs string
    lcs = [""] * (index+1)
 
 
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = n
    j = m
    while i > 0 and j > 0:
 
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if s1[i-1] == s2[j-1]:
            lcs[index-1] = s1[i-1]
            i-=1
            j-=1
            index-=1
 
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i-1][j] > L[i][j-1]:
            i-=1
        else:
            j-=1
    return "".join(lcs)

s1 = "HARRY"
s2 = "SALLY"

assert commonChild(s1, s2) == "AY"
```


```python
"".join(["a", "a"])
```




    'aa'



<h2> Find largest word in dictionary by deleting some characters of given string


```python
def isSubsecuence(str1, str2):
    n = len(str1)
    m = len(str2)
    i = j = 0
    while  i < n and j < m:
        if str1[i] == str2[j]:
            j +=1
        i +=1
    return j == m

def isSubsecuence2(str1, str2):
    m = len(str2)
    j = 0
    for letter in str1:
        if letter == str2[j] and j < m:
            j += 1
    return m == j
    
    
str1 = "apple"
str2 = "ale" 

print isSubsecuence(str1, str2)
print isSubsecuence2(str1, str2)


def findLongestString(words, string):
    
    result = ""
    resultLength = 0
    for word in words:
        
        if resultLength <= len(word) and result != word:
            if isSubsecuence(string, word):
                result = word
                resultLength = len(word)
    return result  
    
l = ["ale", "apple", "monkey", "plea"]
string = "abpcplea"

print findLongestString(l, string)

```

    True
    True
    apple
    


```python
"""solution to the array pair sum problem"""
def array_pair_sum_hash_table(arr, k):
    """
    Use a hash table to store array elements of pairs.
    complexity: O(n)
    """

    result = []
    hash_table = {}

    for e in arr:
        if e in hash_table:
            result.append([k - e, e])
        else:
            hash_table[k - e] = True
    result.reverse()

    return result

```


```python
"""Find subarray with given sum"""
# An efficient program 
# to print subarray
# with sum as given sum 
 
# Returns true if the 
# there is a subarray 
# of arr[] with sum
# equal to 'sum' 
# otherwise returns 
# false. Also, prints 
# the result.
def subArraySum(arr, n, sum):
     
    # Initialize curr_sum as
    # value of first element
    # and starting point as 0 
    curr_sum = arr[0]
    start = 0
 
    # Add elements one by 
    # one to curr_sum and 
    # if the curr_sum exceeds 
    # the sum, then remove 
    # starting element 
    i = 1
    while i <= n:
         
        # If curr_sum exceeds
        # the sum, then remove
        # the starting elements
        while curr_sum > sum and start < i-1:
         
            curr_sum = curr_sum - arr[start]
            start += 1
             
        # If curr_sum becomes
        # equal to sum, then
        # return true
        if curr_sum == sum:
            print ("Sum found between indexes")
            print ("%d and %d"%(start, i-1))
            return 1
 
        # Add this element 
        # to curr_sum
        if i < n:
            curr_sum = curr_sum + arr[i]
        i += 1
 
    # If we reach here, 
    # then no subarray
    print ("No subarray found")
    return 0
 
# Driver program
arr = [15, 2, 4, 8, 9, 5, 10, 23]
n = len(arr)
sum = 23
 
subArraySum(arr, n, sum)
```

    Sum found between indexes
    1 and 4
    




    1




```python
"""Median of two sorted arrays of same size"""

# A Simple Merge based O(n) Python 3 solution 
# to find median of two sorted lists
 
# This function returns median of ar1[] and ar2[].
# Assumptions in this function:
# Both ar1[] and ar2[] are sorted arrays
# Both have n elements
def getMedian( ar1, ar2 , n):
    i = 0 # Current index of i/p list ar1[]
     
    j = 0 # Current index of i/p list ar2[]
     
    m1 = -1
    m2 = -1
     
    # Since there are 2n elements, median
    # will be average of elements at index
    # n-1 and n in the array obtained after
    # merging ar1 and ar2
    count = 0
    while count < n + 1:
        count += 1
         
        # Below is to handle case where all
        # elements of ar1[] are smaller than
        # smallest(or first) element of ar2[]
        if i == n:
            m1 = m2
            m2 = ar2[0]
            break
         
        # Below is to handle case where all 
        # elements of ar2[] are smaller than
        # smallest(or first) element of ar1[]
        elif j == n:
            m1 = m2
            m2 = ar1[0]
            break
        if ar1[i] < ar2[j]:
            m1 = m2 # Store the prev median
            m2 = ar1[i]
            i += 1
        else:
            m1 = m2 # Store the prev median
            m2 = ar2[j]
            j += 1
    return (m1 + m2)/2
 
# Driver code to test above function
ar1 = [1, 12, 15, 26, 38]
ar2 = [2, 13, 17, 30, 45]
n1 = len(ar1)
n2 = len(ar2)
if n1 == n2:
    print("Median is ", getMedian(ar1, ar2, n1))
else:
    print("Doesn't work for arrays of unequal size")
```

<h1> QUESTIONS </h1>

find number of pairs of numbers which add up to a given input<br>
Design LRU cache. <br>
Rotate an array of numbers using just one variable and no additional data structure <br>
Longest substring with no repeating characters <br>
Print characters occurance and duplicate numbers <br>
Write a program to find smallest number in a rotated sorted array.  <br>
Given an array of non-negative numbers & a target value, return the length of smallest subarray whose sum is greater than the target value<br>
From the fibonacci sequence, find the number smaller and greater to a given number. You are given a number not an index.  <br>
given a string, find all Unique substrings with k length, then sort. ex: caaab，k = 2，return aa, ab, ca  <br>
reverse all the vowels in a string  <br>
Reverse a linked list (not just the function).
Given two unsorted arrays, find the median(not the brute force approach). <br> <br>
Group Anagrams

1) Given a list of words, group them by anagrams
Input: List of "cat", "dog", "god"
Output: A Set of Sets of anagrams: {{'cat'}, {'dog', 'god'}}
2) Run this code in the REPL to observe its behaviour. The execution entry point is main().
3) Consider adding some additional tests in doTestsPass().
4) Implement the AnagramSolution group() method correctly.
5) If time permits, try to improve your implementation.  

<br>
Quick Sort, 3 sum question. Leetcode  


```python

```
