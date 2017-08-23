import urllib2

req = urllib2.Request('http://nirgun.caltech.edu:8000/analyze?source=crts&ra=3.220125&dec=5.049086&sr=1000&stats=27')
response = urllib2.urlopen(req)
the_page = response.read()
print the_page
