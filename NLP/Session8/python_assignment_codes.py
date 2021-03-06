# Given a string, find the length of the longest substring without repeating characters.
str = "piyushjain"

def longest_non_repeat(str):   
    i=0
    max_length = 1

    for i,c in enumerate(str):       
        start_at = i
        sub_str=[]         
        while (start_at < len(str)) and (str[start_at] not in sub_str):
            sub_str.append(str[start_at])
            start_at = start_at + 1
        if len(sub_str) > max_length:
            max_length = len(sub_str)
        print(sub_str)
    return max_length

longest_non_repeat(str)

# Given an array of integers, return indices of the two numbers such that they add up to a specific target.
input_array = [2, 7, 11, 15]
target = 26
result = []

for i, num in enumerate(input_array):
    for j in range(i+1, len(input_array)):
        print(i,j)

# Given a sorted integer array without duplicates, return the summary of its ranges.
input_array = [0,1,2,4,5,7]
start=0
result = []
while start < len(input_array):
    end = start 
    while end+1<len(input_array) and ((input_array[end+1] - input_array[end]) == 1):
        end = end + 1
    if end!=start:
        result.append("{0}-->{1}".format(input_array[start], input_array[end]))
        print(result)
    else:
        result.append("{0}".format(input_array[start]))
        print(result)
    start = end+1

print(result)

# Rotate an array of n elements to the right by k steps.
org = [1,2,3,4,5,6,7]
result = org[:]
steps = 3

for idx,num in enumerate(org):
    if idx+steps < len(org):
        result[idx+steps] = org[idx]
    else:
        result[idx+steps-len(org)] = org[idx]

print(result)

# Consider an array of non-negative integers. A second array is formed by shuffling the elements of the first array and deleting a random element. Given these two arrays, find which element is missing in the second array.
first_array = [1,2,3,4,5,6,7]
second_array = [3,7,2,1,4,6]

def finder(first_array, second_array):
    return(sum(first_array) - sum(second_array))

missing_number = finder(first_array, second_array)

print(missing_number)

# Given a collection of intervals which are already sorted by start number, merge all overlapping intervals.
org_intervals = [[1,3],[2,6],[5,10],[11,16],[15,18],[19,22]]
i = 0
while i < len(org_intervals)-1:
    if org_intervals[i+1][0] < org_intervals[i][1]:
        org_intervals[i][1]=org_intervals[i+1][1]
        del org_intervals[i+1]
        i = i - 1
    i = i + 1
print(org_intervals)

# Given a list slice it into a 3 equal chunks and revert each list
sampleList = [11, 45, 8, 23, 14, 12, 78, 45, 89]

length = len(sampleList)
chunkSize  = int(length/3)
start = 0
end = chunkSize
for i in range(1, 4, 1):
  indexes = slice(start, end, 1)
  listChunk = sampleList[indexes]
  mylist = [i for i in listChunk]
  print("After reversing it ", mylist)
  start = end
  if(i != 2):
    end +=chunkSize
  else:
    end += length - chunkSize

# write a program to calculate exponents of an input
input = 9
exponent = 2
final = pow(input, exponent)
print(f'Exponent Value is:{final}')

# write a program to multiply two Matrix 
# 3x3 matrix
X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]
# 3x4 matrix
Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]
# result is 3x4
result = [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]]

# iterate through rows of X
for i in range(len(X)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]
print(f"Final Result is{result}")

# write a program to find and print the remainder of two number

num1 = 12
num2 = 10

ratio = num1 % num2
print(f'remainder:{ratio}')

# reverse a number in Python
number = 1367891
revs_number = 0  
while (number > 0):
  remainder = number % 10
  revs_number = (revs_number * 10) + remainder
  number = number // 10
print("The reverse number is : {}".format(revs_number))

# Python program to compute sum of digits in number
def sumDigits(no):  
    return 0 if no == 0 else int(no % 10) + sumDigits(int(no / 10))   
n = 1234511
print(sumDigits(n))

# Find the middle element of a random number list
my_list = [4,3,2,9,10,44,1]
print("mid value is ",my_list[int(len(my_list)/2)])

# Sort the list in ascending order
my_list = [4,3,2,9,10,44,1]
my_list.sort()
print(f"Ascending Order list:,{my_list}")

# Sort the list in descending order
my_list = [4,3,2,9,10,44,1]
my_list.sort(reverse=True)
print(f"Descending Order list:,{my_list}")

# Concatenation of two List
my_list1 = [4,3,2,9,10,44,1]
my_list2 = [5,6,2,8,15,14,12]
print(f"Sum of two list:,{my_list1+my_list2}")

# Removes the item at the given index from the list and returns the removed item
my_list1 = [4,3,2,9,10,44,1,9,12]
index = 4
print(f"Sum of two list:,{my_list1.pop(index)}")

# Adding Element to a List
animals = ['cat', 'dog', 'rabbit']
animals.append('guinea pig')
print('Updated animals list: ', animals)

# Returns the number of times the specified element appears in the list
vowels = ['a', 'e', 'i', 'o', 'i', 'u']
count = vowels.count('i')
print('The count of i is:', count)

# Count Tuple Elements Inside List
random = ['a', ('a', 'b'), ('a', 'b'), [3, 4]]
count = random.count(('a', 'b'))
print("The count of ('a', 'b') is:", count)

# Removes all items from the list
list = [{1, 2}, ('a'), ['1.1', '2.2']]
list.clear()
print('List:', list)

# access first characters in a string
word = "Hello World"
letter=word[0]
print(f"First Charecter in String:{letter}")

# access Last characters in a string
word = "Hello World"
letter=word[-1]
print(f"First Charecter in String:{letter}")

# Generate a list by list comprehension
list = [x for x in range(10)]
print(f"List Generated by list comprehension:{list}")

# Sort the string list alphabetically

thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(f"Sorted List:{thislist}")

# Join Two Sets
set1 = {"a", "b" , "c"}
set2 = {1, 2, 3}
set3 = set2.union(set1)
print(f"Joined Set:{set3}")

# keep only the items that are present in both sets
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

x.intersection_update(y)
print(f"Duplicate Value in Two set:{x}")

# Keep All items from List But NOT the Duplicates
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}

x.symmetric_difference_update(y)
print(f"Duplicate Value in Two set:{x}")

# Create and print a dictionary
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(f"Sample Dictionary:{thisdict}")

# Calculate the length of dictionary
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

print(f"Length of Dictionary:{len(thisdict)}")

# Evaluate a string and a number
print(bool("Hello"))
print(bool(15))

# Calculate length of a string
word = "Hello World"
print(f"Length of string: {len(word)}")

# Count the number of spaces in a sring
s = "Count, the number of spaces"
lenx = s.count(' ')
print(f"number of spaces in sring: {lenx}")

# Split Strings
word = "Hello World"
ksplit = word.split(' ') 
print(f"Splited Strings: {ksplit}")

# Prints ten dots
ten = "." * 10
print(f"Ten dots: {ten}")

# Replacing a string with another string
word = "Hello World"
replace = "Bye"
input = "Hello"
after_replace = word.replace(input, replace)
print(f"String ater replacement: {after_replace}")

#removes leading characters
word = " xyz "
lstrip = word.lstrip()
print(f"String ater removal of leading characters:{lstrip}")

#removes trailing characters
word = " xyz "
rstrip = word.rstrip()
print(f"String ater removal of trailing characters:{rstrip}")

# check if all char are alphanumeric
word = "Hello World"
check = word.isalnum()
print(f"All char are alphanumeric?:{check}")

# check if all char in the string are alphabetic
word = "Hello World"
check = word.isalpha()
print(f"All char are alphabetic?:{check}")

# test if string contains digits
word = "Hello World"
check = word.isdigit()
print(f"String contains digits?:{check}")

# Test if string contains upper case
word = "Hello World"
check = word.isupper()
print(f"String contains upper case?:{check}")

# Test if string starts with H
word = "Hello World"
check = word.startswith('H')
print(f"String starts with H?:{check}")

# Returns an integer value for the given character
str = "A"
val = ord(str)
print(f"Integer value for the given character?:{val}")

#  Fibonacci series up to 100
n = 100
result = []
a, b = 0 , 1
while b < n:
  result. append( b)
  a, b = b, a + b
final = result
print(f"Fibonacci series up to 100:{final}")

# Counting total Digits in a string
str1 = "abc4234AFde"
digitCount = 0
for i in range(0,len(str1)):
  char = str1[i]
  if(char.isdigit()):
    digitCount += 1
print('Number of digits: ',digitCount)

# Counting total alphanumeric in a string
str1 = "abc4234AFde"
digitCount = 0
for i in range(0,len(str1)):
  char = str1[i]
  if(char.isalpha()):
    digitCount += 1
print('Number of alphanumeric: ',digitCount)

# Counting total Upper Case in a string
str1 = "abc4234AFde"
digitCount = 0
for i in range(0,len(str1)):
  char = str1[i]
  if(char.upper()):
    digitCount += 1
print('Number total Upper Case: ',digitCount)

# Counting total lower Case in a string
str1 = "abc4234AFdeaa"
digitCount = 0
for i in range(0,len(str1)):
  char = str1[i]
  if(char.lower()):
    digitCount += 1
print('Number total lower Case: ',digitCount)

# Bubble sort in python
list1 = [1, 5, 3, 4]

for i in range(len(list1)-1):
  for j in range(i+1,len(list1)):
    if(list1[i] > list1[j]):
      temp = list1[i]
      list1[i] = list1[j]
      list1[j] = temp
print("Bubble Sorted list: ",list1)

# Compute the product of every pair of numbers from two lists
list1 = [1, 2, 3]
list2 = [5, 6, 7] 
final = [a*b for a in list1 for b in list2]
print(f"Product of every pair of numbers from two lists:{final}")

# Calculate the sum of every pair of numbers from two lists
list1 = [1, 2, 3]
list2 = [5, 6, 7] 
final = [a+b for a in list1 for b in list2]
print(f"sum of every pair of numbers from two lists:{final}")

# Calculate the pair-wise product of two lists
list1 = [1, 2, 3]
list2 = [5, 6, 7] 
final = [list1[i]*list2[i] for i in range(len(list1))]
print(f"pair-wise product of two lists:{final}")

# Remove the last element from the stack
s = [1,2,3,4]
print(f"last element from the stack:{s.pop()}")

# Insert a number at the beginning of the queue
q = [1,2,3,4]
q.insert(0,5)
print(f"Revised List:{q}")

# Addition of two vector
v1 = [1,2,3]
v2 = [1,2,3]
s1 = [0,0,0]

for i in range(len(v1)):
  s1[i] = v1[i] + v2[i]
print(f"New Vector:{s1}")

# Replace negative prices with 0 and leave the positive values unchanged in a list
original_prices = [1.25, -9.45, 10.22, 3.78, -5.92, 1.16]
prices = [i if i > 0 else 0 for i in original_prices]
print(f"Final List:{prices}")

# Convert dictionary to JSON
import json
person_dict = {'name': 'Bob',
'age': 12,
'children': None
}
person_json = json.dumps(person_dict)
print(person_json)

# Writing JSON to a file
import json
person_dict = {"name": "Bob",
"languages": ["English", "Fench"],
"married": True,
"age": 32
}
with open('person.txt', 'w') as json_file:
  json.dump(person_dict, json_file)

# Pretty print JSON
import json
person_string = '{"name": "Bob", "languages": "English", "numbers": [2, 1.6, null]}'
person_dict = json.loads(person_string)
print(json.dumps(person_dict, indent = 4, sort_keys=True))

# Check if the key exists or not in JSON
import json

studentJson ="""{
   "id": 1,
   "name": "Piyush Jain",
   "class": null,
   "percentage": 35,
   "email": "piyushjain220@gmail.com"
}"""

print("Checking if percentage key exists in JSON")
student = json.loads(studentJson)
if "percentage" in student:
    print("Key exist in JSON data")
    print(student["name"], "marks is: ", student["percentage"])
else:
    print("Key doesn't exist in JSON data")

# Check if there is a value for a key in JSON
import json

studentJson ="""{
   "id": 1,
   "name": "Piyush Jain",
   "class": null,
   "percentage": 35,
   "email": "piyushjain220@gmail.com"
}"""
student = json.loads(studentJson)
if not (student.get('email') is None):
     print("value is present for given JSON key")
     print(student.get('email'))
else:
    print("value is not present for given JSON key")

# Sort JSON keys in Python and write it into a file
import json
sampleJson = {"id" : 1, "name" : "value2", "age" : 29}

with open("sampleJson.json", "w") as write_file:
    json.dump(sampleJson, write_file, indent=4, sort_keys=True)
print("Done writing JSON data into a file")

#  Given a Python list. Turn every item of a list into its square
aList = [1, 2, 3, 4, 5, 6, 7]
aList =  [x * x for x in aList]
print(aList)

# Remove empty strings from the list of strings
list1 = ["Mike", "", "Emma", "Kelly", "", "Brad"]
resList = [i for i in (filter(None, list1))]
print(resList)

# Write a program which will achieve given a Python list, remove all occurrence of an input from the list
list1 = [5, 20, 15, 20, 25, 50, 20]

def removeValue(sampleList, val):
   return [value for value in sampleList if value != val]
   
resList = removeValue(list1, 20)
print(resList)

#  Generate 3 random integers between 100 and 999 which is divisible by 5
import random

print("Generating 3 random integer number between 100 and 999 divisible by 5")
for num in range(3):
    print(random.randrange(100, 999, 5), end=', ')

# Pick a random character from a given String
import random

name = 'pynative'
char = random.choice(name)
print("random char is ", char)

# Generate  random String of length 5
import random
import string

def randomString(stringLength):
    """Generate a random string of 5 charcters"""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))

print ("Random String is ", randomString(5) )

# Generate a random date between given start and end dates
import random
import time

def getRandomDate(startDate, endDate ):
    print("Printing random date between", startDate, " and ", endDate)
    randomGenerator = random.random()
    dateFormat = '%m/%d/%Y'

    startTime = time.mktime(time.strptime(startDate, dateFormat))
    endTime = time.mktime(time.strptime(endDate, dateFormat))

    randomTime = startTime + randomGenerator * (endTime - startTime)
    randomDate = time.strftime(dateFormat, time.localtime(randomTime))
    return randomDate

print ("Random Date = ", getRandomDate("1/1/2016", "12/12/2018"))

# Write a program which will create a new string by appending s2 in the middle of s1 given two strings, s1 and s2
def appendMiddle(s1, s2):
  middleIndex = int(len(s1) /2)
  middleThree = s1[:middleIndex:]+ s2 +s1[middleIndex:]
  print("After appending new string in middle", middleThree)
  
appendMiddle("Ault", "Kelly")

# Arrange string characters such that lowercase letters should come first
str1 = "PyNaTive"
lower = []
upper = []
for char in str1:
    if char.islower():
        lower.append(char)
    else:
        upper.append(char)
sorted_string = ''.join(lower + upper)
print(sorted_string)

# Given a string, return the sum and average of the digits that appear in the string, ignoring all other characters
import re

inputStr = "English = 78 Science = 83 Math = 68 History = 65"
markList = [int(num) for num in re.findall(r'\b\d+\b', inputStr)]
totalMarks = 0
for mark in markList:
  totalMarks+=mark

percentage = totalMarks/len(markList)  
print("Total Marks is:", totalMarks, "Percentage is ", percentage)

# Given an input string, count occurrences of all characters within a string
str1 = "Apple"
countDict = dict()
for char in str1:
  count = str1.count(char)
  countDict[char]=count
print(countDict)

# Reverse a given string
str1 = "PYnative"
print("Original String is:", str1)

str1 = str1[::-1]
print("Reversed String is:", str1)

# Remove special symbols/Punctuation from a given string
import string

str1 = "/*Jon is @developer & musician"
new_str = str1.translate(str.maketrans('', '', string.punctuation))
print("New string is ", new_str)

# Removal all the characters other than integers from string
str1 = 'I am 25 years and 10 months old'
res = "".join([item for item in str1 if item.isdigit()])
print(res)

# From given string replace each punctuation with #
from string import punctuation

str1 = '/*Jon is @developer & musician!!'
replace_char = '#'
for char in punctuation:
    str1 = str1.replace(char, replace_char)

print("The strings after replacement : ", str1)

# Given a list iterate it and count the occurrence of each element and create a dictionary to show the count of each elemen
sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]
countDict = dict()
for item in sampleList:
  if(item in countDict):
    countDict[item] += 1
  else:
    countDict[item] = 1
  
print("Printing count of each item  ",countDict)

# Given a two list of equal size create a set such that it shows the element from both lists in the pair
firstList = [2, 3, 4, 5, 6, 7, 8]
secondList = [4, 9, 16, 25, 36, 49, 64]
result = zip(firstList, secondList)
resultSet = set(result)
print(resultSet)

# Given a two sets find the intersection and remove those elements from the first set
firstSet  = {23, 42, 65, 57, 78, 83, 29}
secondSet = {57, 83, 29, 67, 73, 43, 48}

intersection = firstSet.intersection(secondSet)
for item in intersection:
  firstSet.remove(item)
print("First Set after removing common element ", firstSet)

# Given a dictionary get all values from the dictionary and add it in a list but don’t add duplicates
speed  ={'jan':47, 'feb':52, 'march':47, 'April':44, 'May':52, 'June':53,
          'july':54, 'Aug':44, 'Sept':54} 

speedList = []
for item in speed.values():
  if item not in speedList:
    speedList.append(item)
print("unique list", speedList)

# Convert decimal number to octal
print('%o,' % (8))

# Convert string into a datetime object
from datetime import datetime
date_string = "Feb 25 2020  4:20PM"
datetime_object = datetime.strptime(date_string, '%b %d %Y %I:%M%p')
print(datetime_object)

# Subtract a week from a given date
from datetime import datetime, timedelta
given_date = datetime(2020, 2, 25)
days_to_subtract = 7
res_date = given_date - timedelta(days=days_to_subtract)
print(res_date)

# Find the day of week of a given date?
from datetime import datetime
given_date = datetime(2020, 7, 26)
print(given_date.strftime('%A'))

#  Add week (7 days) and 12 hours to a given date
from datetime import datetime, timedelta
given_date = datetime(2020, 3, 22, 10, 00, 00)
days_to_add = 7
res_date = given_date + timedelta(days=days_to_add, hours=12)
print(res_date)

# Calculate number of days between two given dates
from datetime import datetime

date_1 = datetime(2020, 2, 25).date()
date_2 = datetime(2020, 9, 17).date()
delta = None
if date_1 > date_2:
    delta = date_1 - date_2
else:
    delta = date_2 - date_1
print("Difference is", delta.days, "days")

# Write a recursive function to calculate the sum of numbers from 0 to 10
def calculateSum(num):
    if num:
        return num + calculateSum(num-1)
    else:
        return 0
res = calculateSum(10)
print(res)

# Generate a Python list of all the even numbers between two given numbers
num1 = 4
num2 = 30
myval = [i for i in range(num1, num2, 2)]
print(myval)

# Return the largest item from the given list
aList = [4, 6, 8, 24, 12, 2]
print(max(aList))

# Write a program to extract each digit from an integer, in the reverse order
number = 7536
while (number > 0):
    digit = number % 10
    number = number // 10
    print(digit, end=" ")

#  Given a Python list, remove all occurrence of a given number from the list
num1 = 20
list1 = [5, 20, 15, 20, 25, 50, 20]

def removeValue(sampleList, val):
   return [value for value in sampleList if value != val]
resList = removeValue(list1, num1)
print(resList)

# Shuffle a list randomly
import random
list = [2,5,8,9,12]
random.shuffle(list)
print ("Printing shuffled list ", list)

# Generate a random n-dimensional array of float numbers
import numpy
random_float_array = numpy.random.rand(2, 2)
print("2 X 2 random float array in [0.0, 1.0] \n", random_float_array,"\n")

# Generate random Universally unique IDs
import uuid
safeId = uuid.uuid4()
print("safe unique id is ", safeId)

# Choose given number of elements from the list with different probability
import random
num1 =5
numberList = [111, 222, 333, 444, 555]
print(random.choices(numberList, weights=(10, 20, 30, 40, 50), k=num1))

# Generate weighted random numbers
import random
randomList = random.choices(range(10, 40, 5), cum_weights=(5, 15, 10, 25, 40, 65), k=6)
print(randomList)

# generating a reliable secure random number
import secrets
print("Random integer number generated using secrets module is ")
number = secrets.randbelow(30)
print(number)

# Calculate memory is being used by an list in Python
import sys
list1 = ['Scott', 'Eric', 'Kelly', 'Emma', 'Smith']
print("size of list = ",sys.getsizeof(list1))

# Find if all elements in a list are identical
listOne = [20, 20, 20, 20]
print("All element are duplicate in listOne:", listOne.count(listOne[0]) == len(listOne))

# Merge two dictionaries in a single expression
currentEmployee = {1: 'Scott', 2: "Eric", 3:"Kelly"}
formerEmployee  = {2: 'Eric', 4: "Emma"}
allEmployee = {**currentEmployee, **formerEmployee}
print(allEmployee)

# Convert two lists into a dictionary
ItemId = [54, 65, 76]
names = ["Hard Disk", "Laptop", "RAM"]
itemDictionary = dict(zip(ItemId, names))
print(itemDictionary)

# Alternate cases in String 
test_str = "geeksforgeeks"
res = "" 
for idx in range(len(test_str)): 
    if not idx % 2 : 
       res = res + test_str[idx].upper() 
    else: 
       res = res + test_str[idx].lower() 
print(res)