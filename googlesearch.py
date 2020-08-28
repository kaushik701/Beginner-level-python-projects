import google
from google import googlesearch 
from googlesearch import search
print(googlesearch.__version__)

query = 'freecodecamp'
for i in search(query,tld = 'com',lang = 'en',num = 10,stop = 10,pause = 2):
	print(i)